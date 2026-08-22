"""Recompute diagnostic OOF headroom above the Public-1159 v82 parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import diagnostics
from src.v53_factorization_offset import apply_offset
from src.v77_team_oof_constrained_blend import single_candidate_headroom
from src.v80_oof_covariance_stack import (
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_late24_diagnostic_candidates,
    _load_parent_axis,
    _archive,
    _assert_target,
)


EPS = 0.001
V82_NAME = "v57_independent_blend_shift"


def add_frozen_shift(
    new_parent: np.ndarray,
    old_parent: np.ndarray,
    old_candidate: np.ndarray,
) -> np.ndarray:
    """Add an already frozen candidate direction above a new exact parent."""

    current = np.asarray(new_parent, dtype=np.float64)
    previous = np.asarray(old_parent, dtype=np.float64)
    candidate = np.asarray(old_candidate, dtype=np.float64)
    if not (
        current.ndim == 1
        and current.shape == previous.shape == candidate.shape
    ):
        raise ValueError("conditional rebase arrays must be aligned vectors")
    if not (
        np.isfinite(current).all()
        and np.isfinite(previous).all()
        and np.isfinite(candidate).all()
    ):
        raise ValueError("conditional rebase arrays contain non-finite values")
    return np.clip(current + candidate - previous, EPS, 1.0 - EPS)


def audit_axis(
    axis_name: str,
    axis: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    old_parent = axis["parent"]
    new_parent = np.asarray(candidates[V82_NAME], dtype=np.float64)
    target = axis["target"]
    frame = pd.DataFrame(
        {
            "target": target,
            "game_month": axis["month"],
            "domain3": axis["domain3"],
        }
    )
    parent_error = new_parent - target
    rows: list[dict[str, Any]] = []
    for name, old_candidate in candidates.items():
        if name == V82_NAME:
            name = "additional_v57_same_direction"
        conditional = add_frozen_shift(new_parent, old_parent, old_candidate)
        raw = diagnostics(
            frame,
            new_parent,
            conditional,
            np.ones(len(frame), dtype=bool),
        )
        headroom = single_candidate_headroom(target, new_parent, conditional)
        candidate_error = conditional - target
        residual_correlation = (
            0.0
            if np.std(parent_error) == 0.0 or np.std(candidate_error) == 0.0
            else float(np.corrcoef(parent_error, candidate_error)[0, 1])
        )
        rows.append(
            {
                "axis": axis_name,
                "candidate_shift": name,
                "raw_gain_above_v82": float(raw["gain"]),
                "raw_positive_month_fraction": float(
                    raw["positive_month_fraction"]
                ),
                "raw_worst_month_gain": float(raw["worst_month_gain"]),
                "raw_minimum_domain_gain": float(raw["minimum_domain_gain"]),
                "conditional_oracle_weight": float(
                    headroom["optimal_candidate_weight"]
                ),
                "conditional_oracle_gain": float(
                    headroom["unclipped_bss_gain"]
                ),
                "residual_correlation_with_v82": residual_correlation,
                "shift_rms": float(headroom["shift_rms"]),
                "diagnostic_only": True,
            }
        )
    return rows


def run(
    project: Path,
    final_parent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = {
        name: _load_parent_axis(final_parent_dir, name)
        for name in ("full_2024", "late_2024")
    }
    common = {
        name: _load_common_candidates(project, axes[name], name)
        for name in axes
    }
    candidates = {
        "full_2024": _load_full24_diagnostic_candidates(
            project, axes["full_2024"], common["full_2024"]
        ),
        "late_2024": _load_late24_diagnostic_candidates(
            project, axes["late_2024"], common["late_2024"]
        ),
    }
    v56 = _archive(
        project
        / "artifacts"
        / "v56_shared_horizon_fm_20260817_01"
        / "outer_full_2024.npz"
    )
    _assert_target(axes["full_2024"]["target"], v56, "v56/full_2024")
    late_mask = axes["full_2024"]["month"] >= 8
    if not np.array_equal(
        axes["full_2024"]["target"][late_mask], axes["late_2024"]["target"]
    ):
        raise ValueError("v56 late target/order mismatch")
    late_frame = pd.DataFrame(
        {
            "target": axes["late_2024"]["target"],
            "game_month": axes["late_2024"]["month"],
            "domain3": axes["late_2024"]["domain3"],
        }
    )
    v56_late, _ = apply_offset(
        late_frame,
        axes["late_2024"]["parent"],
        np.asarray(v56["correction"], dtype=np.float64)[late_mask],
        "F",
        0.10,
    )
    candidates["late_2024"]["v56_shared_horizon_fm_shift"] = v56_late
    for axis_name, registry in candidates.items():
        if V82_NAME not in registry:
            raise ValueError(f"v82 OOF parent missing on {axis_name}")
    rows = [
        row
        for axis_name in axes
        for row in audit_axis(axis_name, axes[axis_name], candidates[axis_name])
    ]
    table = pd.DataFrame(rows).sort_values(
        ["axis", "conditional_oracle_gain"], ascending=[True, False]
    )
    table.to_csv(output_dir / "conditional_headroom.csv", index=False)
    top = {
        axis: table.loc[table["axis"].eq(axis)].iloc[0].to_dict()
        for axis in axes
    }
    result = {
        "protocol": "V83_POST_PUBLIC_1159_CONDITIONAL_HEADROOM_V1",
        "public_parent": 1159.33522,
        "public_gain_vs_1158": 1.2606643248998,
        "oof_parent": "v82 / v57 strict probability blend 10% on R_CORE",
        "top_by_axis": top,
        "candidate_rows": len(table),
        "same_axis_labels_used": True,
        "diagnostic_only": True,
        "eligible_for_packaging": False,
        "public_score_used_for_weight": False,
        "test_csv_read": False,
        "test_aggregate_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.final_parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
