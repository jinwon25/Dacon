"""Strict-forward count-graph partial pooling above v345.

For each pitcher, residual command at the 12 legal ball-strike counts and two
batter hands is estimated with a graph-Laplacian penalty.  Adjacent count
states share strength, and the two batter-hand surfaces are weakly coupled.
All tables use completed-season forward OOF residuals and are averaged over
strictly prior seasons.  Graph strength is selected on full-2022 and
late-2023 before full-2024 is opened once.  Only R_CORE rows are changed.
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
from src.archive.v362_pitcher_situation_lowrank_v345 import (
    SOURCE_YEARS,
    full_row_rms,
    rcore_mask,
)


PROTOCOL = "V365_COUNT_GRAPH_PARTIAL_POOLING_V345_V1"
TARGET = "control_success"
RIDGE = 300.0
GRAPH_STRENGTHS = (30.0, 100.0, 300.0, 1000.0)
HAND_COUPLING = 0.25
WEIGHT = 0.50


def count_hand_position(rows: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(rows["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(rows["strikes_before"], errors="raise").to_numpy(np.int16)
    hand = pd.to_numeric(rows["batter_hand"], errors="raise").to_numpy(np.int16) - 1
    if not (np.isin(balls, range(4)).all() and np.isin(strikes, range(3)).all()):
        raise ValueError("unexpected count")
    if not np.isin(hand, (0, 1)).all():
        raise ValueError("unexpected batter hand")
    return hand * 12 + balls * 3 + strikes


def graph_laplacian() -> np.ndarray:
    adjacency = np.zeros((24, 24), dtype=np.float64)
    for hand in range(2):
        for balls in range(4):
            for strikes in range(3):
                here = hand * 12 + balls * 3 + strikes
                if balls < 3:
                    there = hand * 12 + (balls + 1) * 3 + strikes
                    adjacency[here, there] = adjacency[there, here] = 1.0
                if strikes < 2:
                    there = hand * 12 + balls * 3 + strikes + 1
                    adjacency[here, there] = adjacency[there, here] = 1.0
                other = (1 - hand) * 12 + balls * 3 + strikes
                adjacency[here, other] = adjacency[other, here] = HAND_COUPLING
    return np.diag(adjacency.sum(axis=1)) - adjacency


def fit_model(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    graph_strength: float,
) -> dict[str, Any]:
    position = count_hand_position(rows)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    residual -= float(residual.mean())
    codes, pitcher_ids = pd.factorize(rows["pitcher_id"], sort=True)
    sums = np.zeros((len(pitcher_ids), 24), dtype=np.float64)
    counts = np.zeros((len(pitcher_ids), 24), dtype=np.float64)
    np.add.at(sums, (codes, position), residual)
    np.add.at(counts, (codes, position), 1.0)
    penalty = RIDGE * np.eye(24) + float(graph_strength) * graph_laplacian()
    estimates = np.zeros_like(sums)
    for index in range(len(pitcher_ids)):
        estimates[index] = np.linalg.solve(
            np.diag(counts[index]) + penalty,
            sums[index],
        )
    return {
        "pitcher_ids": np.asarray(pitcher_ids),
        "estimates": estimates,
        "counts": counts,
        "observed_cells": int(np.count_nonzero(counts)),
    }


def map_model(model: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    position = count_hand_position(rows)
    index = pd.Index(model["pitcher_ids"]).get_indexer(rows["pitcher_id"])
    seen = index >= 0
    result = np.zeros(len(rows), dtype=np.float64)
    result[seen] = model["estimates"][index[seen], position[seen]]
    return result


def make_candidate(
    frame: pd.DataFrame, parent: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = rcore_mask(frame) & np.not_equal(signal, 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(output[active] + WEIGHT * signal[active], 0.001, 0.999)
    return output, active


def run(
    train_csv: Path,
    oof_dir: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "batter_hand", "balls_before",
        "strikes_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    season_rows = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (*SOURCE_YEARS, 2024)
    }
    source_axes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in SOURCE_YEARS:
        with np.load(oof_dir / f"wave0_incumbent_validate_{year}.npz", allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        if not np.array_equal(target, season_rows[year][TARGET].to_numpy(np.float64)):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        source_axes[year] = target, parent

    frames = {
        "full_2022": season_rows[2022],
        "late_2023": season_rows[2023].loc[
            season_rows[2023]["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": season_rows[2024],
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64)
            for axis in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )

    all_models: dict[float, dict[int, dict[str, Any]]] = {}
    banks: dict[float, dict[str, np.ndarray]] = {}
    grid: list[dict[str, Any]] = []
    payload: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for strength in GRAPH_STRENGTHS:
        models = {
            year: fit_model(season_rows[year], *source_axes[year], strength)
            for year in SOURCE_YEARS
        }
        all_models[strength] = models
        banks[strength] = {}
        payload[strength] = {}
        record: dict[str, Any] = {"graph_strength": strength}
        for axis, year in (("full_2022", 2022), ("late_2023", 2023), ("full_2024", 2024)):
            prior_years = [value for value in SOURCE_YEARS if value < year]
            signal = np.mean(
                np.vstack([map_model(models[value], frames[axis]) for value in prior_years]),
                axis=0,
            )
            banks[strength][axis] = signal
            if axis != "full_2024":
                candidate, active = make_candidate(frames[axis], parents[axis], signal)
                payload[strength][axis] = candidate, active
                metrics = axis_metrics(frames[axis], parents[axis], candidate, active)
                record[f"{axis}_gain"] = metrics["gain"]
                record[f"{axis}_positive_month_fraction"] = metrics["positive_month_fraction"]
                record[f"{axis}_worst_month_gain"] = metrics["worst_month_gain"]
                record[f"{axis}_rms"] = full_row_rms(parents[axis], candidate)
        record["minimum_source_gain"] = min(
            record["full_2022_gain"], record["late_2023_gain"]
        )
        record["source_pass"] = bool(
            record["full_2022_gain"] > 0.0
            and record["late_2023_gain"] > 0.0
            and record["full_2022_positive_month_fraction"] >= 4.0 / 7.0
            and record["late_2023_positive_month_fraction"] >= 2.0 / 3.0
            and record["full_2022_worst_month_gain"] > -10.0
            and record["late_2023_worst_month_gain"] > -10.0
        )
        grid.append(record)

    table = pd.DataFrame(grid).sort_values(
        ["source_pass", "minimum_source_gain", "full_2022_gain", "late_2023_gain"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(output_dir / "source_grid.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_pass"]]
    restrictions = {
        "official_train_only": True,
        "source_residuals_forward_oof": True,
        "strictly_prior_completed_seasons": True,
        "graph_strength_selected_before_full_2024": True,
        "fixed_ridge_hand_coupling_route_and_dose": True,
        "v345_f_and_team13_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "fixed_recipe": {
                "ridge": RIDGE, "hand_coupling": HAND_COUPLING,
                "weight": WEIGHT, "route": "R_CORE",
            },
            "source_screen": grid,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    strength = float(passing.iloc[0]["graph_strength"])
    signal = banks[strength]["full_2024"]
    candidate, active = make_candidate(frames["full_2024"], parents["full_2024"], signal)
    locked = axis_metrics(frames["full_2024"], parents["full_2024"], candidate, active)
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], candidate)
    delta = candidate - parents["full_2024"]
    union = (np.abs(delta) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(delta[union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidate, active,
        [parents["full_2024"], candidate],
    )
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{axis}": parents[axis] for axis in parents},
        **{
            "candidate_full_2022": payload[strength]["full_2022"][0],
            "candidate_late_2023": payload[strength]["late_2023"][0],
            "candidate_full_2024": candidate,
            "active_full_2024": active,
            "direction_full_2024": signal,
        },
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "fixed_recipe": {
            "ridge": RIDGE, "graph_strength": strength,
            "hand_coupling": HAND_COUPLING, "weight": WEIGHT,
            "route": "R_CORE",
        },
        "source_screen": grid,
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
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.oof_dir, args.v335_axes, args.v345_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
