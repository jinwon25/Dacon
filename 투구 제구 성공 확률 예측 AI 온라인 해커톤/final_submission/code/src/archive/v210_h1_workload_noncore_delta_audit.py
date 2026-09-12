"""Audit the paired workload-H1 delta on non-core competition domains.

Unlike v173, this does not blend the absolute H1 prediction into F or
R_ANCHOR.  It adds only the three-seed prediction difference caused by the new
row-local workload features.  F has two source axes and can be selected;
R_ANCHOR remains diagnostic because full-2022 lacks anchor support.
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
    affine,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness, paired_metrics
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v209_h1_workload_multiseed_audit import (
    SEEDS,
    load_seed_predictions,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V210_H1_WORKLOAD_NONCORE_DELTA_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
SCALES = (0.02, 0.05, 0.10)


def apply_noncore_delta(
    parent: np.ndarray,
    direction: np.ndarray,
    axis: dict[str, np.ndarray],
    domain: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(axis["exact_mask"], dtype=bool)
        & np.asarray(axis["domain3"]).astype(str).__eq__(domain)
    )
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(scale) * np.asarray(direction)[active],
        0.001,
        0.999,
    )
    return output, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["overall_gain"] > 0.0
        and result["positive_month_fraction"] >= 0.50
        and result["worst_month_gain"] > -3.0
        and result["active_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "paired_three_seed_workload_delta_only": True,
        "f_has_two_source_axes": True,
        "r_anchor_is_diagnostic_only": True,
        "source_selection_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    raw_delta: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        baseline, augmented = [], []
        for seed in SEEDS:
            base, aug = load_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            baseline.append(base)
            augmented.append(aug)
        base_h1 = affine(np.mean(baseline, axis=0) + correction[year])
        augmented_h1 = affine(np.mean(augmented, axis=0) + correction[year])
        raw_delta[year] = augmented_h1 - base_h1
    directions = {
        "full_2022": raw_delta[2022],
        "late_2023": raw_delta[2023][late23],
        "full_2024": raw_delta[2024],
    }

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    candidates: dict[float, dict[str, np.ndarray]] = {}
    active_masks: dict[float, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        details[str(scale)], candidates[scale], active_masks[scale] = {}, {}, {}
        row: dict[str, Any] = {"scale": scale}
        for axis_name in AXES:
            candidate, active = apply_noncore_delta(
                parents[axis_name], directions[axis_name], axes[axis_name],
                "F", scale,
            )
            candidates[scale][axis_name] = candidate
            active_masks[scale][axis_name] = active
            result = paired_metrics(
                axes[axis_name], parents[axis_name], candidate, active
            )
            details[str(scale)][axis_name] = result
            row[f"{axis_name}_gain"] = result["overall_gain"]
            row[f"{axis_name}_month_fraction"] = result[
                "positive_month_fraction"
            ]
            row[f"{axis_name}_worst_month"] = result["worst_month_gain"]
        row["source_gate_passed"] = all(
            source_gate(details[str(scale)][axis]) for axis in SOURCE_AXES
        )
        row["source_min_gain"] = min(
            details[str(scale)][axis]["overall_gain"] for axis in SOURCE_AXES
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain"],
        ascending=[False, False], kind="stable",
    )
    table.to_csv(
        output_dir / "f_source_screen.csv", index=False, encoding="utf-8-sig"
    )
    passing = table.loc[table["source_gate_passed"]]

    # R_ANCHOR has only late-2023 source support and can never promote.
    anchor_diagnostic: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        anchor_diagnostic[str(scale)] = {}
        for axis_name in ("late_2023", "full_2024"):
            candidate, active = apply_noncore_delta(
                parents[axis_name], directions[axis_name], axes[axis_name],
                "R_ANCHOR", scale,
            )
            anchor_diagnostic[str(scale)][axis_name] = paired_metrics(
                axes[axis_name], parents[axis_name], candidate, active
            )

    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "f_source_reject",
            "screen": table.to_dict(orient="records"),
            "anchor_diagnostic_only": anchor_diagnostic,
            "eligible_for_packaging": False,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_scale = float(passing.iloc[0]["scale"])
        locked = details[str(selected_scale)]["full_2024"]
        candidate = candidates[selected_scale]["full_2024"]
        active = active_masks[selected_scale]["full_2024"]
        family = [
            candidates[float(scale)]["full_2024"]
            for scale in passing["scale"].tolist()
        ]
        family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate,
            active, family,
        )
        point_pass = bool(
            locked["overall_gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -3.0
            and locked["active_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"],
            candidate=candidate,
            active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "confirmed_f_delta" if point_pass and robust_pass else (
                "f_point_pass_robust_reject" if point_pass else "f_locked_reject"
            ),
            "selected_scale": selected_scale,
            "source": {
                axis: details[str(selected_scale)][axis] for axis in SOURCE_AXES
            },
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "anchor_diagnostic_only": anchor_diagnostic,
            "screen": table.to_dict(orient="records"),
            "eligible_for_joint_audit": bool(point_pass and robust_pass),
            "eligible_for_packaging": False,
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.baseline_checkpoint_dir, args.seed42_dir,
        args.multiseed_dir, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
