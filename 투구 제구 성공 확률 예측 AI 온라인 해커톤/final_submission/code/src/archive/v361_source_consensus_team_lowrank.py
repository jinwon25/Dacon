"""Extend the Public-positive v335 low-rank mechanism to source-stable teams.

Team 13 is already protected by v335/v345.  For every other team, this audit
tests the exact frozen v314 low-rank increment on regular-season rows where
that team is either pitching or batting, excluding any matchup involving team
13.  A team is retained only if full-2022 and late-2023 both have positive
gain, a majority of positive months, and a bounded worst month.  The union of
all passing teams is frozen before full-2024 is evaluated above exact v345.
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


PROTOCOL = "V361_SOURCE_CONSENSUS_TEAM_LOWRANK_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
ANCHOR_TEAM = 13


def team_mask(frame: pd.DataFrame, team: int) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(ANCHOR_TEAM).to_numpy()
        | frame["batter_team_id"].eq(ANCHOR_TEAM).to_numpy()
    )
    involved = (
        frame["pitcher_team_id"].eq(team).to_numpy()
        | frame["batter_team_id"].eq(team).to_numpy()
    )
    return regular & ~anchor & involved


def apply_increment(
    parent: np.ndarray, increment: np.ndarray, active: np.ndarray
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + np.asarray(increment, dtype=np.float64)[active],
        0.001, 0.999,
    )
    return output


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    v314_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season", "game_month", "game_type", "pitcher_id", "batter_id",
            "pitcher_team_id", "batter_team_id", TARGET,
        ],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        v335 = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        if not np.array_equal(saved["parent_full_2024"], v335["full_2024"]):
            raise ValueError("v345/v335 parent mismatch")
        v345 = saved["candidate_full_2024"].astype(np.float64)
    with np.load(v314_axes, allow_pickle=False) as saved:
        increments = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            - saved[f"parent_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }

    source_origins = ("full_2022", "late_2023")
    common_teams = set(
        pd.concat(
            [
                frames[source_origins[0]]["pitcher_team_id"],
                frames[source_origins[0]]["batter_team_id"],
            ]
        ).dropna().astype(int).unique()
    )
    for origin in source_origins[1:]:
        teams = set(
            pd.concat(
                [frames[origin]["pitcher_team_id"], frames[origin]["batter_team_id"]]
            ).dropna().astype(int).unique()
        )
        common_teams.intersection_update(teams)
    common_teams.discard(ANCHOR_TEAM)

    team_results: list[dict[str, Any]] = []
    passing: list[int] = []
    for team in sorted(common_teams):
        per_origin: dict[str, Any] = {}
        for origin in source_origins:
            active = team_mask(frames[origin], team)
            candidate = apply_increment(v335[origin], increments[origin], active)
            per_origin[origin] = axis_metrics(
                frames[origin], v335[origin], candidate, active
            )
        passed = bool(
            per_origin["full_2022"]["gain"] > 0.0
            and per_origin["late_2023"]["gain"] > 0.0
            and per_origin["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
            and per_origin["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and per_origin["full_2022"]["worst_month_gain"] > -10.0
            and per_origin["late_2023"]["worst_month_gain"] > -10.0
        )
        if passed:
            passing.append(team)
        team_results.append(
            {"team": team, "source_gate_passed": passed, "metrics": per_origin}
        )

    parents = {
        "full_2022": v335["full_2022"],
        "late_2023": v335["late_2023"],
        "full_2024": v345,
    }
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        active[origin] = np.zeros(len(frames[origin]), dtype=bool)
        for team in passing:
            active[origin] |= team_mask(frames[origin], team)
        candidates[origin] = apply_increment(
            parents[origin], increments[origin], active[origin]
        )
        metrics[origin] = axis_metrics(
            frames[origin], parents[origin], candidates[origin], active[origin]
        )

    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = (
        _robustness(
            axes24, parents["full_2024"], candidates["full_2024"],
            active["full_2024"], [parents["full_2024"], candidates["full_2024"]],
        )
        if active["full_2024"].any()
        else {"not_run": "no team passed the source gate"}
    )
    source_pass = bool(
        passing
        and metrics["full_2022"]["gain"] > 0.0
        and metrics["late_2023"]["gain"] > 0.0
    )
    locked = metrics["full_2024"]
    eligible = bool(
        source_pass
        and locked["gain"] >= 1.5
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and robustness.get("pitcher", {}).get("p05", -np.inf) > 0.0
        and robustness.get("chronological_block", {}).get("p05", -np.inf) > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{f"candidate_{origin}": candidates[origin] for origin in ORIGINS},
        **{f"active_{origin}": active[origin] for origin in ORIGINS},
        **{f"increment_{origin}": increments[origin] for origin in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if eligible else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "recipe": {
            "expert": "frozen v314/v50 lowrank_s300_r2 at dose 0.50",
            "team_rule": "union of all non-13 teams passing both source origins",
            "selected_teams": passing,
            "anchor_team_excluded": ANCHOR_TEAM,
        },
        "team_source_results": team_results,
        "union_metrics": metrics,
        "locked_robustness": robustness,
        "source_gate_passed": source_pass,
        "eligible_for_packaging": eligible,
        "restrictions": {
            "official_train_only": True,
            "frozen_lowrank_model_and_dose": True,
            "team_selection_full2022_and_late2023_only": True,
            "all_passing_teams_used_no_best_team_selection": True,
            "v345_f_and_team13_rows_preserved_exactly": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
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
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--v314-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.v335_axes, args.v345_axes, args.v314_axes,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
