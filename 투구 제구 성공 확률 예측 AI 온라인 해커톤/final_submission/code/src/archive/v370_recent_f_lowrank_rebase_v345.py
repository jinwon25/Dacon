"""One-shot rebase of the frozen v313 recent-F low-rank direction above v345.

The v313 recipe and dose are not changed.  Late-2023 is the sole source gate;
full-2024 is opened only if the original stability requirements still pass
after changing the parent to v345.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V370_RECENT_F_LOWRANK_REBASE_V345_V1"
TARGET = "control_success"


def run(
    train_csv: Path,
    v313_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "asof_pitcher_n", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v313_axes, allow_pickle=False) as saved:
        directions = {
            axis: saved[f"direction_{axis}"].astype(np.float64)
            for axis in frames
        }
        historical_active = {
            axis: saved[f"active_{axis}"].astype(bool)
            for axis in frames
        }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            "late_2023": saved["candidate_late_2023"].astype(np.float64)
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )

    def build(axis: str) -> tuple[np.ndarray, np.ndarray]:
        candidate = np.clip(parents[axis] + directions[axis], 0.001, 0.999)
        active = historical_active[axis] & (np.abs(candidate - parents[axis]) > 1e-15)
        return candidate, active

    source_candidate, source_active = build("late_2023")
    source = axis_metrics(
        frames["late_2023"], parents["late_2023"], source_candidate, source_active
    )
    source["full_row_rms_shift"] = full_row_rms(
        parents["late_2023"], source_candidate
    )
    source_pass = bool(
        source["gain"] > 0.0
        and source["positive_month_fraction"] >= 2.0 / 3.0
        and source["worst_month_gain"] > -15.0
        and source["full_row_rms_shift"] >= 0.0003
    )
    restrictions = {
        "official_train_only": True,
        "v313_recipe_dose_direction_and_mask_frozen": True,
        "one_source_gate_only": True,
        "full2024_opened_only_after_source_gate": True,
        "one_shot_family_audit": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not source_pass:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_late_2023": source,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    locked_candidate, locked_active = build("full_2024")
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], locked_candidate, locked_active
    )
    locked["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], locked_candidate
    )
    increment = locked_candidate - parents["full_2024"]
    union = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = (
        float(np.corrcoef(increment[union], v345_increment[union])[0, 1])
        if int(union.sum()) > 2
        and float(np.std(increment[union])) > 0.0
        and float(np.std(v345_increment[union])) > 0.0
        else 0.0
    )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], locked_candidate, locked_active,
        [parents["full_2024"], locked_candidate],
    )
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -15.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=source_candidate,
        active_late_2023=source_active,
        direction_late_2023=directions["late_2023"],
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        direction_full_2024=directions["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "source_late_2023": source,
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v313-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v313_axes, args.v335_axes, args.v345_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
