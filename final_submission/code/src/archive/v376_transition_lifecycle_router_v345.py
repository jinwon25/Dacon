"""Targeted lifecycle router for the frozen v345 player-transition increment.

The v345 transition correction is uniform on R_CORE.  This audit asks whether
its dose should differ only by fixed baseball lifecycle groups: current career
support and prior-season roster transition status.  Every group must prefer
the same alternative dose on full-2022 and late-2023 before it can alter the
exact v345 incumbent on locked full-2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics, entity_transition_state
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V376_TRANSITION_LIFECYCLE_ROUTER_V345_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
SCHEMAS = ("support", "status", "status_support")
MULTIPLIERS = (0.0, 0.5, 1.0, 1.5)
MIN_ROWS = 2000


def lifecycle_groups(train: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    state = entity_transition_state(
        train, rows, year, "pitcher_id", "pitcher_team_id"
    )
    support_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0)
    support = pd.cut(
        support_n,
        bins=[-np.inf, 100.0, 800.0, 2000.0, np.inf],
        labels=["LOW", "DEVELOPING", "ESTABLISHED", "WORKHORSE"],
    ).astype(str)
    status = state["status"].astype(str).reset_index(drop=True)
    return pd.DataFrame({
        "support": support.reset_index(drop=True),
        "status": status,
        "status_support": status + "|" + support.reset_index(drop=True),
    })


def candidate(
    current: np.ndarray,
    transition: np.ndarray,
    group_values: np.ndarray,
    multipliers: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(len(current), dtype=np.float64)
    for group, multiplier in multipliers.items():
        scale[group_values == group] = float(multiplier)
    delta = (scale - 1.0) * np.asarray(transition, dtype=np.float64)
    changed = np.abs(delta) > 1e-15
    out = np.asarray(current, dtype=np.float64).copy()
    out[changed] = np.clip(out[changed] + delta[changed], 0.001, 0.999)
    return out, changed


def run(
    train_csv: Path,
    v338_axes: Path,
    beta_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "asof_pitcher_n", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    years = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    groups = {
        axis: lifecycle_groups(train, frames[axis], years[axis]) for axis in ORIGINS
    }
    with np.load(v338_axes, allow_pickle=False) as saved:
        v335 = {axis: saved[f"parent_{axis}"].astype(np.float64) for axis in ORIGINS}
        transition = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) - v335[axis]
            for axis in ORIGINS
        }
    with np.load(beta_axes, allow_pickle=False) as saved:
        for axis in ORIGINS:
            if not np.array_equal(saved[f"parent_{axis}"], v335[axis]):
                raise ValueError(f"Beta parent mismatch: {axis}")
        beta = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) - v335[axis]
            for axis in ORIGINS
        }
    source_current = {
        axis: np.clip(v335[axis] + transition[axis] + beta[axis], 0.001, 0.999)
        for axis in ("full_2022", "late_2023")
    }
    with np.load(v345_axes, allow_pickle=False) as saved:
        if not np.array_equal(saved["parent_full_2024"], v335["full_2024"]):
            raise ValueError("v345 parent mismatch")
        incumbent24 = saved["candidate_full_2024"].astype(np.float64)

    schema_results: list[dict[str, Any]] = []
    schema_payload: dict[str, tuple[dict[str, float], dict[str, np.ndarray]]] = {}
    for schema in SCHEMAS:
        common = set(groups["full_2022"][schema].unique()) & set(
            groups["late_2023"][schema].unique()
        )
        mapping: dict[str, float] = {}
        group_audit: list[dict[str, Any]] = []
        for group in sorted(common):
            counts = {
                axis: int(np.count_nonzero(
                    (groups[axis][schema].to_numpy() == group)
                    & (np.abs(transition[axis]) > 1e-15)
                ))
                for axis in ("full_2022", "late_2023")
            }
            if min(counts.values()) < MIN_ROWS:
                continue
            options: list[dict[str, Any]] = []
            for multiplier in MULTIPLIERS:
                per_axis: dict[str, Any] = {}
                for axis in ("full_2022", "late_2023"):
                    group_mask = groups[axis][schema].to_numpy() == group
                    proposal = source_current[axis].copy()
                    delta = (multiplier - 1.0) * transition[axis]
                    active_group = group_mask & (np.abs(transition[axis]) > 1e-15)
                    proposal[active_group] = np.clip(
                        proposal[active_group] + delta[active_group], 0.001, 0.999
                    )
                    per_axis[axis] = axis_metrics(
                        frames[axis], source_current[axis], proposal, active_group
                    )
                options.append({
                    "multiplier": multiplier,
                    "minimum_gain": min(
                        per_axis["full_2022"]["gain"], per_axis["late_2023"]["gain"]
                    ),
                    "metrics": per_axis,
                })
            eligible = [
                option for option in options
                if option["multiplier"] != 1.0
                and option["metrics"]["full_2022"]["gain"] > 0.0
                and option["metrics"]["late_2023"]["gain"] > 0.0
                and option["metrics"]["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
                and option["metrics"]["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
                and option["metrics"]["full_2022"]["worst_month_gain"] > -10.0
                and option["metrics"]["late_2023"]["worst_month_gain"] > -10.0
            ]
            if eligible:
                selected = max(eligible, key=lambda value: (value["minimum_gain"], -abs(value["multiplier"] - 1.0)))
                mapping[group] = float(selected["multiplier"])
            else:
                selected = next(value for value in options if value["multiplier"] == 1.0)
            group_audit.append({
                "group": group, "counts": counts,
                "selected_multiplier": float(selected["multiplier"]),
                "options": options,
            })
        if not mapping:
            schema_results.append({
                "schema": schema, "mapping": {}, "group_audit": group_audit,
                "source_metrics": {}, "minimum_source_gain": 0.0,
                "source_pass": False,
            })
            schema_payload[schema] = ({}, {
                "full_2022": source_current["full_2022"].copy(),
                "late_2023": source_current["late_2023"].copy(),
            })
            continue
        source_candidates: dict[str, np.ndarray] = {}
        source_metrics: dict[str, Any] = {}
        for axis in ("full_2022", "late_2023"):
            value, changed = candidate(
                source_current[axis], transition[axis],
                groups[axis][schema].to_numpy(), mapping,
            )
            source_candidates[axis] = value
            source_metrics[axis] = axis_metrics(
                frames[axis], source_current[axis], value, changed
            )
        minimum_gain = min(
            source_metrics["full_2022"]["gain"], source_metrics["late_2023"]["gain"]
        )
        passed = bool(
            mapping
            and source_metrics["full_2022"]["gain"] > 0.0
            and source_metrics["late_2023"]["gain"] > 0.0
            and source_metrics["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
            and source_metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and source_metrics["full_2022"]["worst_month_gain"] > -10.0
            and source_metrics["late_2023"]["worst_month_gain"] > -10.0
        )
        schema_results.append({
            "schema": schema, "mapping": mapping, "group_audit": group_audit,
            "source_metrics": source_metrics, "minimum_source_gain": minimum_gain,
            "source_pass": passed,
        })
        schema_payload[schema] = (mapping, source_candidates)

    passing = sorted(
        (row for row in schema_results if row["source_pass"]),
        key=lambda row: (row["minimum_source_gain"], -SCHEMAS.index(row["schema"])),
        reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "fixed_lifecycle_groups_and_four_doses": True,
        "each_changed_group_positive_on_both_sources": True,
        "source_selection_full2022_and_late2023_only": True,
        "full2024_opened_only_after_source_selection": True,
        "only_frozen_transition_increment_rescaled": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "schema_results": schema_results, "locked_origin_opened": False,
            "eligible_for_packaging": False, "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    schema = str(chosen["schema"])
    mapping = {str(key): float(value) for key, value in chosen["mapping"].items()}
    candidate24, changed24 = candidate(
        incumbent24, transition["full_2024"],
        groups["full_2024"][schema].to_numpy(), mapping,
    )
    locked = axis_metrics(frames["full_2024"], incumbent24, candidate24, changed24)
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
        parent_full_2022=source_current["full_2022"],
        candidate_full_2022=schema_payload[schema][1]["full_2022"],
        parent_late_2023=source_current["late_2023"],
        candidate_late_2023=schema_payload[schema][1]["late_2023"],
        parent_full_2024=incumbent24, candidate_full_2024=candidate24,
        transition_increment_full_2024=transition["full_2024"],
        active_full_2024=changed24,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if eligible else "locked_reject",
        "schema_results": schema_results,
        "selected_source_recipe": {"schema": schema, "mapping": mapping},
        "locked_full_2024_vs_v345": locked, "locked_robustness": robustness,
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
