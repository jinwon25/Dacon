"""Source-selected dose audit for the Public-positive v345 components.

This is not a global probability rescale.  It changes only the already-frozen
player-transition and Beta-cell increments while preserving the workload,
Futures and team-13 heads.  Multipliers are selected on full-2022 and
late-2023 OOF analogues; full-2024 above exact v345 is opened once afterward.
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


PROTOCOL = "V375_CHAMPION_COMPONENT_DOSE_V345_V1"
TARGET = "control_success"
TRANSITION_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
BETA_MULTIPLIERS = (0.00, 0.50, 1.00, 1.50, 2.00)
ORIGINS = ("full_2022", "late_2023", "full_2024")


def scaled(
    parent: np.ndarray,
    transition: np.ndarray,
    beta: np.ndarray,
    transition_multiplier: float,
    beta_multiplier: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + float(transition_multiplier) * np.asarray(transition, dtype=np.float64)
        + float(beta_multiplier) * np.asarray(beta, dtype=np.float64),
        0.001,
        0.999,
    )


def run(
    train_csv: Path,
    v338_axes: Path,
    beta_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v338_axes, allow_pickle=False) as saved:
        parents = {axis: saved[f"parent_{axis}"].astype(np.float64) for axis in ORIGINS}
        transition = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) - parents[axis]
            for axis in ORIGINS
        }
        transition_active = {
            axis: saved[f"active_{axis}"].astype(bool) for axis in ORIGINS
        }
    with np.load(beta_axes, allow_pickle=False) as saved:
        for axis in ORIGINS:
            if not np.array_equal(saved[f"parent_{axis}"], parents[axis]):
                raise ValueError(f"v338/Beta parent mismatch: {axis}")
        beta = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) - parents[axis]
            for axis in ORIGINS
        }
        beta_active = {axis: saved[f"active_{axis}"].astype(bool) for axis in ORIGINS}
    with np.load(v345_axes, allow_pickle=False) as saved:
        if not np.array_equal(saved["parent_full_2024"], parents["full_2024"]):
            raise ValueError("v345/v335 parent mismatch")
        incumbent24 = saved["candidate_full_2024"].astype(np.float64)

    trials: list[dict[str, Any]] = []
    payload: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for tm in TRANSITION_MULTIPLIERS:
        for bm in BETA_MULTIPLIERS:
            key = (tm, bm)
            payload[key] = {}
            row: dict[str, Any] = {
                "transition_multiplier": tm,
                "transition_absolute_dose": 0.25 * tm,
                "beta_multiplier": bm,
                "beta_absolute_dose": 0.10 * bm,
            }
            for axis in ("full_2022", "late_2023"):
                candidate = scaled(parents[axis], transition[axis], beta[axis], tm, bm)
                active = transition_active[axis] | beta_active[axis]
                payload[key][axis] = candidate
                row[axis] = axis_metrics(frames[axis], parents[axis], candidate, active)
            row["minimum_source_gain"] = min(
                row["full_2022"]["gain"], row["late_2023"]["gain"]
            )
            row["source_pass"] = bool(
                row["full_2022"]["gain"] > 0.0
                and row["late_2023"]["gain"] > 0.0
                and row["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
                and row["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
                and row["full_2022"]["worst_month_gain"] > -10.0
                and row["late_2023"]["worst_month_gain"] > -10.0
            )
            trials.append(row)
    passing = sorted(
        (row for row in trials if row["source_pass"]),
        key=lambda row: (
            row["minimum_source_gain"],
            row["full_2022"]["gain"] + row["late_2023"]["gain"],
            -abs(row["transition_multiplier"] - 1.0) - abs(row["beta_multiplier"] - 1.0),
        ),
        reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "two_frozen_champion_increment_families_only": True,
        "source_selection_full2022_and_late2023_only": True,
        "full2024_opened_only_after_source_selection": True,
        "futures_anchor_and_workload_preserved_exactly": True,
        "not_global_probability_scaling": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "source_trials": trials, "locked_origin_opened": False,
            "eligible_for_packaging": False, "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    tm = float(chosen["transition_multiplier"])
    bm = float(chosen["beta_multiplier"])
    candidate24 = np.clip(
        incumbent24 + (tm - 1.0) * transition["full_2024"]
        + (bm - 1.0) * beta["full_2024"],
        0.001,
        0.999,
    )
    changed24 = np.abs(candidate24 - incumbent24) > 1e-15
    locked = axis_metrics(
        frames["full_2024"], incumbent24, candidate24, changed24
    )
    locked["full_row_rms_shift"] = full_row_rms(incumbent24, candidate24)
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, incumbent24, candidate24, changed24, [incumbent24, candidate24]
    )
    eligible = bool(
        locked["gain"] >= 1.5
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.00035
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[(tm, bm)]["full_2022"],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[(tm, bm)]["late_2023"],
        parent_full_2024=incumbent24, candidate_full_2024=candidate24,
        transition_increment_full_2024=transition["full_2024"],
        beta_increment_full_2024=beta["full_2024"],
        active_full_2024=changed24,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if eligible else "locked_reject",
        "source_trials": trials, "selected_source_recipe": chosen,
        "locked_full_2024_vs_v345": locked,
        "locked_robustness": robustness,
        "eligible_for_packaging": eligible, "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v338-axes", type=Path, required=True)
    parser.add_argument("--beta-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v338_axes, args.beta_axes, args.v345_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
