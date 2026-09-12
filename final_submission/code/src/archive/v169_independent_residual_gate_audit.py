"""Audit independent model residual directions above the exact current contract.

Existing v155/v156 results compared independent CatBoost/DCNv2 predictions to
an older common parent.  This diagnostic reuses their frozen forward OOF
predictions but changes only the comparison parent: exact H1/C3/affine source
analogues for 2022/late-2023 and the reconstructed deployed v167 contract for
2024.  No model is refit and no test row is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    C3_BASE_RECENT_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    c3_mix,
    compose,
    gate_library,
    metrics,
    rcore_mask,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V169_INDEPENDENT_RESIDUAL_GATE_AUDIT_V1"
ETA_GRID = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15)


def optimal_eta(
    target: np.ndarray,
    base: np.ndarray,
    direct: np.ndarray,
    active: np.ndarray,
) -> float:
    direction = np.asarray(direct, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    denominator = float(np.sum(np.square(direction[active])))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(residual[active] * direction[active]) / denominator)


def apply_direction(
    base: np.ndarray,
    direct: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active]
        + float(eta)
        * (np.asarray(direct, dtype=np.float64)[active] - output[active]),
        0.001,
        0.999,
    )
    return output


def _source_bases(
    axes: dict[str, dict[str, np.ndarray]], source: dict[str, dict[str, Any]]
) -> dict[str, np.ndarray]:
    output = {}
    for name, values in source.items():
        c3 = c3_mix(values["sign"], values["recent"], C3_BASE_RECENT_WEIGHT)
        output[name] = compose(
            values["component"],
            values["h1"],
            c3,
            axes[name],
            h1_weight=H1_BASE_WEIGHT,
        )
    return output


def _load_directs(
    catboost_paths: list[Path], dcnv2_path: Path, late23: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    families: dict[str, dict[str, np.ndarray]] = {}
    for path in catboost_paths:
        with np.load(path, allow_pickle=False) as saved:
            families[path.parent.name] = {
                "full_2022": saved["catboost_full_2022"].astype(np.float64),
                "late_2023": saved["catboost_full_2023"].astype(np.float64)[late23],
                "full_2024": saved["catboost_full_2024"].astype(np.float64),
            }
    with np.load(dcnv2_path, allow_pickle=False) as saved:
        families[dcnv2_path.parent.name] = {
            "full_2022": saved["dcnv2_full_2022"].astype(np.float64),
            "late_2023": saved["dcnv2_full_2023"].astype(np.float64)[late23],
            "full_2024": saved["dcnv2_full_2024"].astype(np.float64),
        }
    return families


def _mask_library(
    frame: pd.DataFrame,
    base: np.ndarray,
    direct: np.ndarray,
) -> dict[str, np.ndarray]:
    baseball = gate_library()
    difference = np.asarray(direct, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    masks = {
        name: np.asarray(gate(frame), dtype=bool)
        for name, gate in baseball.items()
        if name
        in {
            "all_rcore",
            "runners",
            "high_li",
            "runners_or_high_li",
            "pressure_count",
            "runners_or_pressure_count",
            "late_close",
        }
    }
    masks.update(
        {
            "direct_above": difference > 0.0,
            "direct_below": difference < 0.0,
            "abs_disagreement_ge_002": np.abs(difference) >= 0.02,
            "abs_disagreement_ge_005": np.abs(difference) >= 0.05,
            "direct_above_ge_002": difference >= 0.02,
            "direct_below_le_m002": difference <= -0.02,
        }
    )
    return masks


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
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    bases = _source_bases(axes, source)
    locked_values, parity = _locked_contract(
        axes["full_2024"],
        frames[2024],
        correction[2024],
        h1_path,
        c3_path,
        v160_path,
    )
    bases["full_2024"] = locked_values["current"]
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    directs = _load_directs(catboost_paths, dcnv2_path, late23)
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for family, family_direct in directs.items():
        details[family] = {}
        masks_by_axis = {
            name: _mask_library(axis_frames[name], bases[name], family_direct[name])
            for name in axes
        }
        for mask_name in masks_by_axis["full_2022"]:
            details[family][mask_name] = {}
            for eta in ETA_GRID:
                axis_results = {}
                for axis_name in axes:
                    exact_active = axes[axis_name]["exact_mask"].astype(bool)
                    active = (
                        rcore_mask(axes[axis_name])
                        & masks_by_axis[axis_name][mask_name]
                    )
                    candidate = apply_direction(
                        bases[axis_name], family_direct[axis_name], active, eta
                    )
                    result = metrics(axes[axis_name], bases[axis_name], candidate)
                    result["active_rows"] = int(active.sum())
                    result["active_exact_rows"] = int((active & exact_active).sum())
                    axis_results[axis_name] = result
                details[family][mask_name][str(eta)] = axis_results
                rows.append(
                    {
                        "family": family,
                        "mask": mask_name,
                        "eta": eta,
                        "source_2022_gain": axis_results["full_2022"]["gain"],
                        "source_2022_worst_month": axis_results["full_2022"][
                            "worst_month_gain"
                        ],
                        "confirm_late_2023_gain": axis_results["late_2023"]["gain"],
                        "confirm_late_2023_worst_month": axis_results["late_2023"][
                            "worst_month_gain"
                        ],
                        "locked_2024_gain": axis_results["full_2024"]["gain"],
                        "locked_2024_worst_month": axis_results["full_2024"][
                            "worst_month_gain"
                        ],
                        "locked_2024_positive_month_fraction": axis_results[
                            "full_2024"
                        ]["positive_month_fraction"],
                        "locked_active_rows": axis_results["full_2024"][
                            "active_rows"
                        ],
                    }
                )

        details[family]["unconstrained_optimal_eta"] = {}
        for axis_name in axes:
            active = rcore_mask(axes[axis_name]) & axes[axis_name]["exact_mask"].astype(bool)
            details[family]["unconstrained_optimal_eta"][axis_name] = optimal_eta(
                axes[axis_name]["target"],
                bases[axis_name],
                family_direct[axis_name],
                active,
            )

    table = pd.DataFrame(rows)
    table["source_min_gain"] = table[
        ["source_2022_gain", "confirm_late_2023_gain"]
    ].min(axis=1)
    table = table.sort_values(
        ["source_min_gain", "locked_2024_gain"], ascending=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "candidate_audit.csv", index=False, encoding="utf-8-sig")
    result = {
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "current_contract_parity": parity,
        "eta_grid": list(ETA_GRID),
        "details": details,
        "limitations": [
            "The independent OOF models were trained by an earlier experiment and are not packaged model artifacts.",
            "2024 has been repeatedly inspected and is development-contaminated.",
            "No candidate is eligible for packaging from this screen alone.",
        ],
        "restrictions": {
            "external_2025_outcomes_used": False,
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
        args.train_csv,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.catboost_path,
        args.dcnv2_path,
        args.output_dir,
    )
    print(json.dumps({
        family: values["unconstrained_optimal_eta"]
        for family, values in result["details"].items()
    }, ensure_ascii=False, indent=2))
    table = pd.read_csv(args.output_dir / "candidate_audit.csv", encoding="utf-8-sig")
    print(table.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
