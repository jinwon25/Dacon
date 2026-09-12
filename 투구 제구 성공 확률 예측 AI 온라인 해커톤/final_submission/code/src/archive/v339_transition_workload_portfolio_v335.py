"""Combine the frozen v338 transition and v209 workload-H1 increments.

The three-seed v209 direction is the only workload component eligible for the
portfolio.  The earlier single-seed v206 direction is reported as a diagnostic
but cannot be selected from this repeated full-2024 audit.
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


PROTOCOL = "V339_TRANSITION_WORKLOAD_PORTFOLIO_V335_V1"
TARGET = "control_success"


def add_frozen_increment(
    parent: np.ndarray,
    component_parent: np.ndarray,
    component_candidate: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    delta = np.asarray(component_candidate, dtype=np.float64) - np.asarray(
        component_parent, dtype=np.float64
    )
    output[active] = np.clip(output[active] + delta[active], 0.001, 0.999)
    return output


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        parent, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def run(
    train_csv: Path,
    v335_axes: Path,
    v338_axes: Path,
    v206_axis: Path,
    v206_summary: Path,
    v209_axis: Path,
    v209_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    frame = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    with np.load(v335_axes, allow_pickle=False) as saved:
        parent = saved["candidate_full_2024"].astype(np.float64)
    with np.load(v338_axes, allow_pickle=False) as saved:
        transition = saved["candidate_full_2024"].astype(np.float64)
        transition_active = saved["active_full_2024"].astype(bool)

    components: dict[str, dict[str, np.ndarray]] = {}
    for name, path in (("v206_single_seed", v206_axis), ("v209_three_seed", v209_axis)):
        with np.load(path, allow_pickle=False) as saved:
            components[name] = {
                "parent": saved["parent"].astype(np.float64),
                "candidate": saved["candidate"].astype(np.float64),
                "active": saved["active"].astype(bool),
            }

    axes = {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    candidates: dict[str, np.ndarray] = {"v338_transition": transition}
    active: dict[str, np.ndarray] = {"v338_transition": transition_active}
    for name, values in components.items():
        candidates[name] = add_frozen_increment(
            parent,
            values["parent"],
            values["candidate"],
            values["active"],
        )
        active[name] = values["active"]

    selected_component = "v209_three_seed"
    selected_values = components[selected_component]
    candidates["portfolio"] = add_frozen_increment(
        transition,
        selected_values["parent"],
        selected_values["candidate"],
        selected_values["active"],
    )
    active["portfolio"] = transition_active | selected_values["active"]

    metrics: dict[str, Any] = {}
    for name, candidate in candidates.items():
        metrics[name] = axis_metrics(frame, parent, candidate, active[name])
        metrics[name]["full_row_rms_shift"] = full_row_rms(parent, candidate)

    transition_delta = transition - parent
    workload_delta = candidates[selected_component] - parent
    nonzero = (np.abs(transition_delta) > 1e-15) | (np.abs(workload_delta) > 1e-15)
    correlation = float(
        np.corrcoef(transition_delta[nonzero], workload_delta[nonzero])[0, 1]
    )
    robustness = _robustness(
        axes,
        parent,
        candidates["portfolio"],
        active["portfolio"],
        [parent, transition, candidates[selected_component], candidates["portfolio"]],
    )
    portfolio = metrics["portfolio"]
    passed = bool(
        portfolio["gain"] > metrics["v338_transition"]["gain"]
        and portfolio["gain"] >= 2.0
        and portfolio["positive_month_fraction"] >= 0.625
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent,
        candidate_full_2024=candidates["portfolio"],
        active_full_2024=active["portfolio"],
        transition_increment_full_2024=transition_delta,
        workload_increment_full_2024=workload_delta,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "inventory_candidate" if passed else "inventory_reject",
        "selected_workload_component": selected_component,
        "selection_reason": "three paired seeds; v206 is diagnostic only",
        "metrics": metrics,
        "increment_correlation": correlation,
        "support": {
            "transition_rows": int(transition_active.sum()),
            "workload_rows": int(selected_values["active"].sum()),
            "overlap_rows": int(
                np.count_nonzero(transition_active & selected_values["active"])
            ),
        },
        "locked_robustness": robustness,
        "rms_goal_context": {
            "locked_full_row_rms": portfolio["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": float(
                portfolio["full_row_rms_shift"] / 0.004551
            ),
        },
        "candidate_gate_passed": passed,
        "historical_source_evidence": {
            "v206": json.loads(v206_summary.read_text(encoding="utf-8"))["source"],
            "v209": json.loads(v209_summary.read_text(encoding="utf-8"))["source"],
        },
        "selection_warning": (
            "the portfolio is evaluated on an already exposed full-2024 axis; "
            "source arrays for a v335-parent portfolio were not retained"
        ),
        "restrictions": {
            "official_train_only": True,
            "frozen_v338_transition": True,
            "frozen_three_seed_v209_workload": True,
            "v206_cannot_promote": True,
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
    parser.add_argument("--v338-axes", type=Path, required=True)
    parser.add_argument("--v206-axis", type=Path, required=True)
    parser.add_argument("--v206-summary", type=Path, required=True)
    parser.add_argument("--v209-axis", type=Path, required=True)
    parser.add_argument("--v209-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v338_axes,
        args.v206_axis,
        args.v206_summary,
        args.v209_axis,
        args.v209_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
