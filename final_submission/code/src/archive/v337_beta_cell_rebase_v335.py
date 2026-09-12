"""Audit the frozen v328 consensus Beta-Binomial cell above v335.

The cell ``R_CORE|DEVELOPING|MIXED`` and the v321 10% Beta-Binomial
direction were selected using the two source origins in v328.  This module
does not refit or retune them.  Full-2024 is a repeated development audit,
not a fresh holdout, because v328 already reported that origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import (
    axis_metrics,
    build_archetypes,
    schema_key,
)


PROTOCOL = "V337_BETA_CELL_REBASE_V335_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
SCHEMA = ("route", "support", "pitchmix")
FROZEN_GROUP = "R_CORE|DEVELOPING|MIXED"


def fixed_cell_mask(frame: pd.DataFrame) -> np.ndarray:
    return schema_key(build_archetypes(frame), SCHEMA) == FROZEN_GROUP


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(candidate, dtype=np.float64)
                    - np.asarray(parent, dtype=np.float64)
                )
            )
        )
    )


def run(
    train_csv: Path,
    v335_axes: Path,
    v318_axes: Path,
    beta_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "balls_before",
        "strikes_before", "pitcher_hand", "batter_hand", "li", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        anchor_increment24 = saved["candidate_full_2024"].astype(np.float64) - saved[
            "parent_full_2024"
        ].astype(np.float64)
    with np.load(v318_axes, allow_pickle=False) as saved:
        futures_increment24 = saved["candidate_full_2024"].astype(np.float64) - saved[
            "parent_full_2024"
        ].astype(np.float64)
    with np.load(beta_axes, allow_pickle=False) as saved:
        directions = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            - saved[f"baseline_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }

    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        active[origin] = fixed_cell_mask(frames[origin])
        candidates[origin] = parents[origin].copy()
        candidates[origin][active[origin]] = np.clip(
            parents[origin][active[origin]] + directions[origin][active[origin]],
            0.001,
            0.999,
        )
        metrics[origin] = axis_metrics(
            frames[origin], parents[origin], candidates[origin], active[origin]
        )
        metrics[origin]["full_row_rms_shift"] = full_row_rms(
            parents[origin], candidates[origin]
        )

    frame24 = frames["full_2024"]
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
        candidates["full_2024"],
        active["full_2024"],
        [parents["full_2024"], candidates["full_2024"]],
    )
    increment24 = candidates["full_2024"] - parents["full_2024"]
    support = {
        "active_rows": int(active["full_2024"].sum()),
        "overlap_with_v320_f_increment": int(
            np.count_nonzero(
                (np.abs(increment24) > 1e-15)
                & (np.abs(futures_increment24) > 1e-15)
            )
        ),
        "overlap_with_v335_anchor_increment": int(
            np.count_nonzero(
                (np.abs(increment24) > 1e-15)
                & (np.abs(anchor_increment24) > 1e-15)
            )
        ),
    }
    locked = metrics["full_2024"]
    passed = bool(
        all(metrics[name]["gain"] > 0.0 for name in ORIGINS)
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in ORIGINS},
        **{f"candidate_{name}": candidates[name] for name in ORIGINS},
        **{f"active_{name}": active[name] for name in ORIGINS},
        **{f"direction_{name}": directions[name] for name in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "frozen_recipe": {
            "source": "v328 consensus router",
            "schema": list(SCHEMA),
            "group": FROZEN_GROUP,
            "expert": "v321 strict Beta-Binomial 10% direction",
            "route_or_dose_retuned": False,
        },
        "metrics": metrics,
        "locked_robustness": robustness,
        "support_orthogonality": support,
        "rms_goal_context": {
            "locked_full_row_rms": locked["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": float(
                locked["full_row_rms_shift"] / 0.004551
            ),
        },
        "locked_gate_passed": passed,
        "selection_warning": (
            "full-2024 is a repeated development audit because v328 already "
            "reported this origin; it is not an independent holdout"
        ),
        "restrictions": {
            "official_train_only": True,
            "fixed_source_selected_cell": True,
            "fixed_beta_direction": True,
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
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--beta-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v318_axes,
        args.beta_axes,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
