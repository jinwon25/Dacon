"""Rebase the frozen v293 F population direction on v335 and v339.

The v293 rule was selected on late-2023 without IDs.  This audit preserves its
active rows and exact per-row increment, then asks whether that increment is
still useful above the deployed v335 F specialist.  Because v339 only changes
R_CORE rows, the combined portfolio should have disjoint support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V341_F_POPULATION_PORTFOLIO_V335_V1"
TARGET = "control_success"


def add_frozen_increment(
    parent: np.ndarray,
    old_parent: np.ndarray,
    old_candidate: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Add a frozen component only on its originally selected rows."""
    parent = np.asarray(parent, dtype=np.float64)
    old_parent = np.asarray(old_parent, dtype=np.float64)
    old_candidate = np.asarray(old_candidate, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if not (parent.shape == old_parent.shape == old_candidate.shape == active.shape):
        raise ValueError("all arrays must have identical shapes")
    output = parent.copy()
    increment = old_candidate - old_parent
    output[active] = np.clip(output[active] + increment[active], 0.001, 0.999)
    return output


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        parent, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def run(
    train_csv: Path,
    v335_axes: Path,
    v339_axes: Path,
    v293_axes: Path,
    v293_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season",
            "game_month",
            "game_type",
            "pitcher_id",
            "batter_id",
            TARGET,
        ],
        low_memory=False,
    )
    frames = {
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64)
            for axis in frames
        }
    with np.load(v293_axes, allow_pickle=False) as saved:
        old_parents = {
            axis: saved[f"parent_{axis}"].astype(np.float64) for axis in frames
        }
        old_candidates = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) for axis in frames
        }
        actives = {
            axis: saved[f"active_{axis}"].astype(bool) for axis in frames
        }
    with np.load(v339_axes, allow_pickle=False) as saved:
        v339_parent = saved["parent_full_2024"].astype(np.float64)
        v339_candidate = saved["candidate_full_2024"].astype(np.float64)
        v339_active = saved["active_full_2024"].astype(bool)
    if np.max(np.abs(v339_parent - parents["full_2024"])) > 1e-12:
        raise ValueError("v339 parent is not v335")

    component_candidates: dict[str, np.ndarray] = {}
    component_metrics: dict[str, Any] = {}
    for axis, frame in frames.items():
        candidate = add_frozen_increment(
            parents[axis],
            old_parents[axis],
            old_candidates[axis],
            actives[axis],
        )
        component_candidates[axis] = candidate
        component_metrics[axis] = axis_metrics(
            frame, parents[axis], candidate, actives[axis]
        )
        component_metrics[axis]["full_row_rms_shift"] = full_row_rms(
            parents[axis], candidate
        )

    portfolio = add_frozen_increment(
        v339_candidate,
        old_parents["full_2024"],
        old_candidates["full_2024"],
        actives["full_2024"],
    )
    portfolio_active = v339_active | actives["full_2024"]
    frame24 = frames["full_2024"]
    v339_metrics = axis_metrics(
        frame24, parents["full_2024"], v339_candidate, v339_active
    )
    v339_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], v339_candidate
    )
    portfolio_metrics = axis_metrics(
        frame24, parents["full_2024"], portfolio, portfolio_active
    )
    portfolio_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], portfolio
    )

    overlap_rows = int(np.count_nonzero(v339_active & actives["full_2024"]))
    if overlap_rows:
        raise ValueError("v339 and v293 support unexpectedly overlap")
    axes24 = {
        "target": frame24[TARGET].to_numpy(np.float64),
        "game_month": frame24["game_month"].to_numpy(np.int16),
        "pitcher_id": frame24["pitcher_id"].to_numpy(),
        "batter_id": frame24["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame24), dtype=bool),
    }
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        portfolio,
        portfolio_active,
        [
            parents["full_2024"],
            v339_candidate,
            component_candidates["full_2024"],
            portfolio,
        ],
    )
    passed = bool(
        component_metrics["late_2023"]["gain"] > 0.0
        and component_metrics["full_2024"]["gain"] > 0.0
        and portfolio_metrics["gain"] > v339_metrics["gain"]
        and portfolio_metrics["positive_month_fraction"] >= 0.75
        and overlap_rows == 0
    )
    f_increment = component_candidates["full_2024"] - parents["full_2024"]
    v339_increment = v339_candidate - parents["full_2024"]
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=portfolio,
        active_full_2024=portfolio_active,
        f_population_increment_full_2024=f_increment,
        v339_increment_full_2024=v339_increment,
    )
    prior_summary = json.loads(v293_summary.read_text(encoding="utf-8"))
    summary = {
        "protocol": PROTOCOL,
        "status": "inventory_candidate" if passed else "inventory_reject",
        "recipe": {
            "parent": "v335",
            "portfolio_parent": "v339 transition + three-seed workload",
            "new_component": "v293 frozen consensus-add ID-free LGB F direction",
            "rule_or_dose_retuned": False,
        },
        "component_rebase_metrics": component_metrics,
        "v339_metrics": v339_metrics,
        "portfolio_metrics": portfolio_metrics,
        "support": {
            "v339_rows": int(v339_active.sum()),
            "f_population_rows": int(actives["full_2024"].sum()),
            "overlap_rows": overlap_rows,
            "f_mask_purity": float(
                frame24.loc[actives["full_2024"], "game_type"].astype(str).eq("F").mean()
            ),
        },
        "locked_robustness": robustness,
        "rms_goal_context": {
            "locked_full_row_rms": portfolio_metrics["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": float(
                portfolio_metrics["full_row_rms_shift"] / 0.004551
            ),
        },
        "candidate_gate_passed": passed,
        "historical_v293_selected": prior_summary["selected"],
        "selection_warning": (
            "v293 used late-2023 to select its rule and full-2024 has already "
            "been exposed; this is a fixed inventory rebase, not a fresh holdout"
        ),
        "restrictions": {
            "official_train_only": True,
            "v293_rule_and_dose_frozen": True,
            "v339_components_frozen": True,
            "test_csv_read": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v339-axes", type=Path, required=True)
    parser.add_argument("--v293-axes", type=Path, required=True)
    parser.add_argument("--v293-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v339_axes,
        args.v293_axes,
        args.v293_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
