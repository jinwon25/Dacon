"""Audit frozen independent OOF directions in every competition domain.

The deployed v167 row-region overlays only change ``R_CORE``.  This audit therefore
compares the already-frozen v155/v156 forward predictions with the exact
current parent separately in ``R_CORE``, ``R_ANCHOR`` and ``F``.  It never
fits a model or reads test data.  ``R_ANCHOR`` has no 2022 support and is
reported as a one-source diagnostic, never as an eligible recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    _load_year_context,
    _locked_contract,
    _source_contracts,
    metrics,
)
from src.archive.v169_independent_residual_gate_audit import (
    ETA_GRID,
    _load_directs,
    _source_bases,
    apply_direction,
    optimal_eta,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V171_INDEPENDENT_DOMAIN_RESIDUAL_AUDIT_V1"
DOMAINS = ("R_CORE", "R_ANCHOR", "F")


def domain_mask(axis: dict[str, np.ndarray], domain: str) -> np.ndarray:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}")
    return axis["domain3"].astype(str) == domain


def contextual_masks(
    frame: pd.DataFrame,
    base: np.ndarray,
    direct: np.ndarray,
) -> dict[str, np.ndarray]:
    difference = np.asarray(direct, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    runners = frame["num_runners_on"].to_numpy(np.float64) > 0.0
    high_li = frame["li"].to_numpy(np.float64) >= 1.5
    pressure_count = (
        frame["balls_before"].to_numpy(np.int16) >= 2
    ) | (frame["strikes_before"].to_numpy(np.int16) >= 2)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "runners_or_high_li": runners | high_li,
        "runners_or_pressure_count": runners | pressure_count,
        "abs_disagreement_ge_002": np.abs(difference) >= 0.02,
        "abs_disagreement_ge_005": np.abs(difference) >= 0.05,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    catboost_paths: list[Path],
    dcnv2_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(axes, frames, correction, v104_path, h1_path, c3_path)
    bases = _source_bases(axes, source)
    locked, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024], h1_path, c3_path,
        v160_path,
    )
    bases["full_2024"] = locked["current"]
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    directs = _load_directs(catboost_paths, dcnv2_path, late23)

    rows: list[dict[str, Any]] = []
    optimal: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for family, family_direct in directs.items():
        optimal[family] = {}
        coverage[family] = {}
        masks = {
            axis_name: contextual_masks(
                axis_frames[axis_name], bases[axis_name], family_direct[axis_name]
            )
            for axis_name in axes
        }
        for domain in DOMAINS:
            optimal[family][domain] = {}
            coverage[family][domain] = {}
            for axis_name in axes:
                active = domain_mask(axes[axis_name], domain) & axes[axis_name][
                    "exact_mask"
                ].astype(bool)
                coverage[family][domain][axis_name] = int(active.sum())
                optimal[family][domain][axis_name] = (
                    optimal_eta(
                        axes[axis_name]["target"], bases[axis_name],
                        family_direct[axis_name], active,
                    )
                    if active.any()
                    else None
                )
            for gate_name in masks["full_2022"]:
                for eta in ETA_GRID:
                    results: dict[str, dict[str, Any]] = {}
                    for axis_name in axes:
                        active = (
                            domain_mask(axes[axis_name], domain)
                            & masks[axis_name][gate_name]
                        )
                        candidate = apply_direction(
                            bases[axis_name], family_direct[axis_name], active, eta
                        )
                        result = metrics(
                            axes[axis_name], bases[axis_name], candidate
                        )
                        result["active_rows"] = int(active.sum())
                        results[axis_name] = result
                    source_supported = all(
                        coverage[family][domain][name] > 0
                        for name in ("full_2022", "late_2023")
                    )
                    source_gains = [
                        results[name]["gain"]
                        for name in ("full_2022", "late_2023")
                        if coverage[family][domain][name] > 0
                    ]
                    rows.append({
                        "family": family,
                        "domain": domain,
                        "gate": gate_name,
                        "eta": eta,
                        "source_supported": source_supported,
                        "source_min_gain": min(source_gains) if source_gains else np.nan,
                        "source_2022_gain": results["full_2022"]["gain"],
                        "confirm_late_2023_gain": results["late_2023"]["gain"],
                        "locked_2024_gain": results["full_2024"]["gain"],
                        "locked_2024_worst_month": results["full_2024"][
                            "worst_month_gain"
                        ],
                        "locked_2024_positive_month_fraction": results[
                            "full_2024"
                        ]["positive_month_fraction"],
                        "locked_active_rows": results["full_2024"]["active_rows"],
                    })

    table = pd.DataFrame(rows).sort_values(
        ["source_supported", "source_min_gain", "locked_2024_gain"],
        ascending=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "candidate_audit.csv", index=False, encoding="utf-8-sig")
    eligible = table[
        table["source_supported"]
        & (table["source_min_gain"] > 0.0)
        & (table["locked_2024_gain"] > 0.0)
    ]
    result = {
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "current_contract_parity": parity,
        "coverage": coverage,
        "unconstrained_optimal_eta": optimal,
        "eligible_diagnostic_count": int(len(eligible)),
        "best_eligible_diagnostics": eligible.head(30).to_dict("records"),
        "limitations": [
            "Frozen OOF predictions are available, but their fitted runtime checkpoints are not packaged here.",
            "R_ANCHOR has no 2022 support and cannot pass the two-source gate.",
            "2024 is a repeatedly inspected development axis, not a pristine selector.",
        ],
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--catboost-path", type=Path, action="append", required=True)
    parser.add_argument("--dcnv2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.catboost_path,
        args.dcnv2_path, args.output_dir,
    )
    print(json.dumps({
        "eligible_diagnostic_count": result["eligible_diagnostic_count"],
        "optimal": result["unconstrained_optimal_eta"],
        "best": result["best_eligible_diagnostics"][:15],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
