"""Source-consensus platoon partial pooling above the Public v345 parent.

The frozen v311 pitcher-by-batter-hand direction failed when applied globally.
This audit protects the Public-positive F and R_ANCHOR routes and considers only
two predeclared baseball schemas inside R_CORE.  A cell is retained only when
the same frozen direction improves both full-2022 and late-2023.  The schema is
selected without reading full-2024, which is opened once as a development
audit above the exact v345 OOF analogue.
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


PROTOCOL = "V346_PLATOON_CONSENSUS_REBASE_V345_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
SOURCE_ORIGINS = ("full_2022", "late_2023")
SCHEMAS = (
    ("route", "platoon_state"),
    ("route", "support", "platoon_state"),
)
MIN_GROUP_ROWS = 750


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


def apply_groups(
    parent: np.ndarray,
    direction: np.ndarray,
    keys: np.ndarray,
    groups: set[str],
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(eligible, dtype=bool) & np.isin(keys, sorted(groups))
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + np.asarray(direction, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output, active


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    platoon_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "balls_before",
        "strikes_before", "li", TARGET,
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
            for origin in SOURCE_ORIGINS
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    with np.load(platoon_axes, allow_pickle=False) as saved:
        directions = {
            origin: (
                saved[f"candidate_{origin}"].astype(np.float64)
                - saved[f"parent_{origin}"].astype(np.float64)
            )
            for origin in ORIGINS
        }
        covered = {
            origin: saved[f"active_{origin}"].astype(bool)
            for origin in ORIGINS
        }

    archetypes = {origin: build_archetypes(frames[origin]) for origin in ORIGINS}
    rcore = {
        origin: archetypes[origin]["route"].eq("R_CORE").to_numpy()
        for origin in ORIGINS
    }
    eligible = {
        origin: rcore[origin] & covered[origin]
        for origin in ORIGINS
    }

    schema_results: list[dict[str, Any]] = []
    schema_payload: dict[str, dict[str, Any]] = {}
    for schema in SCHEMAS:
        name = "+".join(schema)
        keys = {
            origin: schema_key(archetypes[origin], schema)
            for origin in ORIGINS
        }
        common_groups = sorted(
            set(np.unique(keys["full_2022"][eligible["full_2022"]]))
            & set(np.unique(keys["late_2023"][eligible["late_2023"]]))
        )
        retained: set[str] = set()
        group_screen: list[dict[str, Any]] = []
        for group in common_groups:
            source_metrics: dict[str, Any] = {}
            sufficient = True
            for origin in SOURCE_ORIGINS:
                group_mask = eligible[origin] & (keys[origin] == group)
                if int(group_mask.sum()) < MIN_GROUP_ROWS:
                    sufficient = False
                    break
                candidate, active = apply_groups(
                    parents[origin], directions[origin], keys[origin],
                    {group}, eligible[origin],
                )
                source_metrics[origin] = axis_metrics(
                    frames[origin], parents[origin], candidate, active
                )
            if not sufficient:
                continue
            passed = all(
                source_metrics[origin]["gain"] > 0.0
                and source_metrics[origin]["positive_month_fraction"] >= 0.5
                for origin in SOURCE_ORIGINS
            )
            if passed:
                retained.add(group)
            group_screen.append({
                "group": group,
                "retained": passed,
                "source": source_metrics,
            })

        union_metrics: dict[str, Any] = {}
        union_candidates: dict[str, np.ndarray] = {}
        union_active: dict[str, np.ndarray] = {}
        for origin in SOURCE_ORIGINS:
            candidate, active = apply_groups(
                parents[origin], directions[origin], keys[origin], retained,
                eligible[origin],
            )
            union_candidates[origin] = candidate
            union_active[origin] = active
            union_metrics[origin] = axis_metrics(
                frames[origin], parents[origin], candidate, active
            )
        minimum_gain = min(
            union_metrics[origin]["gain"] for origin in SOURCE_ORIGINS
        )
        mean_gain = float(np.mean([
            union_metrics[origin]["gain"] for origin in SOURCE_ORIGINS
        ]))
        result = {
            "schema": name,
            "complexity": len(schema),
            "retained_groups": sorted(retained),
            "retained_group_count": len(retained),
            "minimum_source_gain": minimum_gain,
            "mean_source_gain": mean_gain,
            "source": union_metrics,
            "group_screen": group_screen,
        }
        schema_results.append(result)
        schema_payload[name] = {
            "schema": schema,
            "keys": keys,
            "groups": retained,
            "candidates": union_candidates,
            "active": union_active,
        }

    passing = [item for item in schema_results if item["retained_group_count"] > 0]
    if not passing:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "schema_results": schema_results,
            "candidate_gate_passed": False,
            "restrictions": {
                "official_train_only": True,
                "frozen_v311_direction_and_dose": True,
                "f_and_r_anchor_protected": True,
                "schema_and_cells_selected_without_full_2024": True,
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

    selected = max(
        passing,
        key=lambda item: (
            item["minimum_source_gain"], item["mean_source_gain"],
            -item["complexity"],
        ),
    )
    payload = schema_payload[selected["schema"]]
    locked_candidate, locked_active = apply_groups(
        parents["full_2024"], directions["full_2024"],
        payload["keys"]["full_2024"], payload["groups"],
        eligible["full_2024"],
    )
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], locked_candidate,
        locked_active,
    )
    locked["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], locked_candidate
    )
    increment = locked_candidate - parents["full_2024"]
    nonzero = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(increment[nonzero], v345_increment[nonzero])[0, 1])
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
        locked["gain"] > 0.5
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
        and locked["full_row_rms_shift"] >= 0.0003
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        direction_full_2024=directions["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "selected_schema": selected,
        "schema_results": schema_results,
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "candidate_gate_passed": passed,
        "selection_warning": (
            "full-2024 is development-contaminated and was opened only after "
            "the source-consensus cells and schema were fixed"
        ),
        "restrictions": {
            "official_train_only": True,
            "frozen_v311_direction_and_dose": True,
            "f_and_r_anchor_protected": True,
            "schema_and_cells_selected_without_full_2024": True,
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
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--platoon-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.v335_axes, args.v345_axes, args.platoon_axes,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
