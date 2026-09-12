"""Screen public independent forward-OOF predictions above the exact row-region parent.

This is a research-only screen.  Public model artifacts are never copied into a
submission package.  A family is worth reproducing locally only when a frozen
small convex blend improves late-2023 first and then survives the untouched
2024 contract.  Test rows and Public scores are not read.
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
    metrics,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.core.contract import _load_contract_axis


PROTOCOL = "V181_PUBLIC_INDEPENDENT_OOF_SCREEN_V1"
FAMILIES = (
    "hierarchical_stack",
    "adaptive_gate",
    "psych_latent",
    "psych_regime_film",
)
ROUTES = ("ALL", "R_CORE", "F")
WEIGHTS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)


def convex_blend(
    parent: np.ndarray,
    independent: np.ndarray,
    active: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    proposal = (
        (1.0 - float(weight)) * output
        + float(weight) * np.asarray(independent, dtype=np.float64)
    )
    output[np.asarray(active, dtype=bool)] = proposal[np.asarray(active, dtype=bool)]
    return np.clip(output, 0.001, 0.999)


def route_mask(axis: dict[str, np.ndarray], route: str) -> np.ndarray:
    exact = np.asarray(axis["exact_mask"], dtype=bool)
    domain = axis["domain3"].astype(str)
    if route == "ALL":
        return exact
    if route not in {"R_CORE", "F"}:
        raise ValueError(f"unknown route: {route}")
    return exact & np.equal(domain, route)


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def locked_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.625
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def load_public_family(
    root: Path,
    family: str,
    late23: np.ndarray,
    truth23: np.ndarray,
    truth24: np.ndarray,
    pitcher23: np.ndarray,
    pitcher24: np.ndarray,
) -> dict[str, np.ndarray]:
    path = root / f"{family}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["y23"].astype(np.int8), truth23.astype(np.int8)):
            raise ValueError(f"2023 label alignment failed: {family}")
        if not np.array_equal(saved["y24"].astype(np.int8), truth24.astype(np.int8)):
            raise ValueError(f"2024 label alignment failed: {family}")
        if not np.array_equal(saved["pitcher23"].astype(np.int64), pitcher23):
            raise ValueError(f"2023 pitcher alignment failed: {family}")
        if not np.array_equal(saved["pitcher24"].astype(np.int64), pitcher24):
            raise ValueError(f"2024 pitcher alignment failed: {family}")
        return {
            "late_2023": saved["p23"].astype(np.float64)[late23],
            "full_2024": saved["p24"].astype(np.float64),
        }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    public_oof_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    axes = {
        # exact_parent_parents reconstructs both historical source contracts as a
        # consistency check even though this screen selects on late-2023 only.
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes,
        raw_frames,
        correction,
        v104_path,
        h1_path,
        c3_path,
        v160_path,
        bridge_oof,
    )
    truth23 = raw_frames[2023]["control_success"].to_numpy(np.int8)
    truth24 = raw_frames[2024]["control_success"].to_numpy(np.int8)
    pitcher23 = raw_frames[2023]["pitcher_id"].to_numpy(np.int64)
    pitcher24 = raw_frames[2024]["pitcher_id"].to_numpy(np.int64)

    source_rows: list[dict[str, Any]] = []
    public_predictions: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        external = load_public_family(
            public_oof_root,
            family,
            late23,
            truth23,
            truth24,
            pitcher23,
            pitcher24,
        )
        public_predictions[family] = external
        for route in ROUTES:
            active = route_mask(axes["late_2023"], route)
            for weight in WEIGHTS:
                candidate = convex_blend(
                    parents["late_2023"], external["late_2023"], active, weight
                )
                result = metrics(axes["late_2023"], parents["late_2023"], candidate)
                source_rows.append(
                    {
                        "family": family,
                        "route": route,
                        "weight": weight,
                        **result,
                        "source_gate_passed": source_gate(result),
                    }
                )

    source = pd.DataFrame(source_rows)
    source["robust_source_score"] = source[["gain", "worst_month_gain"]].min(axis=1)
    ranking = source.sort_values(
        ["source_gate_passed", "robust_source_score", "gain", "weight"],
        ascending=[False, False, False, True],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_screen.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]].copy()
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        # The winning recipe is frozen using late-2023 only.  Full-2024 is
        # opened once for the selected recipe; it cannot choose another arm.
        selected = passing.iloc[0]
        family = str(selected["family"])
        route = str(selected["route"])
        weight = float(selected["weight"])
        active24 = route_mask(axes["full_2024"], route)
        candidate24 = convex_blend(
            parents["full_2024"],
            public_predictions[family]["full_2024"],
            active24,
            weight,
        )
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        robust = _robustness(
            axes["full_2024"],
            parents["full_2024"],
            candidate24,
            active24,
            [candidate24, parents["full_2024"]],
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        eligible = bool(locked_gate(locked) and robust_pass)
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"],
            candidate=candidate24,
            independent=public_predictions[family]["full_2024"],
            active=active24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "research_signal_pass" if eligible else "locked_reject",
            "selected_on_late_2023_only": {
                "family": family,
                "route": route,
                "weight": weight,
            },
            "source": selected.to_dict(),
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": locked_gate(locked),
            "robust_gate_passed": robust_pass,
            "eligible_for_local_reproduction": eligible,
            "eligible_for_packaging": False,
            "packaging_blocker": "public artifacts are research evidence only; reproduce the model locally",
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_labels_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_candidate": True,
        "public_artifact_in_submission": False,
        "research_only_until_local_reproduction": True,
        "locked_2024_development_contaminated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--public-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.public_oof_root,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
