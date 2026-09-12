"""Temporal aggregation variants for the v365 count-graph signal.

Four baseball-motivated, fixed aggregations of strictly prior pitcher-season
count-hand surfaces are screened on full-2022 and late-2023: recency decay,
season sample reliability, their product, and latest observed season.  Only a
source-passing aggregation is opened on full-2024, once, above v345 R_CORE.
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
from src.archive.v362_pitcher_situation_lowrank_v345 import SOURCE_YEARS, full_row_rms
from src.archive.v365_count_graph_partial_pooling_v345 import (
    count_hand_position,
    fit_model,
    make_candidate,
)


PROTOCOL = "V367_COUNT_GRAPH_TEMPORAL_AGGREGATION_V345_V1"
TARGET = "control_success"
GRAPH_STRENGTH = 300.0
SCHEMES = ("recency", "reliability", "recency_reliability", "latest")


def map_value_support(model: dict[str, Any], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    position = count_hand_position(rows)
    index = pd.Index(model["pitcher_ids"]).get_indexer(rows["pitcher_id"])
    seen = index >= 0
    value = np.zeros(len(rows), dtype=np.float64)
    support = np.zeros(len(rows), dtype=np.float64)
    value[seen] = model["estimates"][index[seen], position[seen]]
    support[seen] = model["counts"][index[seen]].sum(axis=1)
    return value, support


def aggregate_signal(
    models: dict[int, dict[str, Any]], rows: pd.DataFrame, audit_year: int, scheme: str
) -> np.ndarray:
    years = [year for year in SOURCE_YEARS if year < audit_year]
    values, supports = zip(*(map_value_support(models[year], rows) for year in years))
    value = np.vstack(values)
    support = np.vstack(supports)
    seen = support > 0.0
    age = np.asarray([audit_year - year - 1 for year in years], dtype=np.float64)[:, None]
    recency = np.power(0.5, age)
    reliability = np.minimum(1.0, support / 500.0)
    if scheme == "recency":
        weight = seen * recency
    elif scheme == "reliability":
        weight = seen * reliability
    elif scheme == "recency_reliability":
        weight = seen * recency * reliability
    elif scheme == "latest":
        weight = np.zeros_like(value)
        for column in range(len(rows)):
            valid = np.flatnonzero(seen[:, column])
            if len(valid):
                weight[valid[-1], column] = 1.0
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    denominator = weight.sum(axis=0)
    result = np.zeros(len(rows), dtype=np.float64)
    active = denominator > 0.0
    result[active] = (weight[:, active] * value[:, active]).sum(axis=0) / denominator[active]
    return result


def run(train_csv: Path, oof_dir: Path, v335_axes: Path, v345_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season", "game_month", "game_type", "pitcher_id", "batter_id",
            "pitcher_team_id", "batter_team_id", "batter_hand", "balls_before",
            "strikes_before", TARGET,
        ],
        low_memory=False,
    )
    seasons = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (*SOURCE_YEARS, 2024)
    }
    models: dict[int, dict[str, Any]] = {}
    for year in SOURCE_YEARS:
        with np.load(oof_dir / f"wave0_incumbent_validate_{year}.npz", allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        if not np.array_equal(target, seasons[year][TARGET].to_numpy(np.float64)):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        models[year] = fit_model(seasons[year], target, parent, GRAPH_STRENGTH)

    frames = {
        "full_2022": seasons[2022],
        "late_2023": seasons[2023].loc[seasons[2023]["game_month"].ge(8)].reset_index(drop=True),
        "full_2024": seasons[2024],
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = saved["candidate_full_2024"].astype(np.float64) - saved["parent_full_2024"].astype(np.float64)

    grid: list[dict[str, Any]] = []
    payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for scheme in SCHEMES:
        payload[scheme] = {}
        record: dict[str, Any] = {"scheme": scheme}
        for name, year in (("full_2022", 2022), ("late_2023", 2023)):
            signal = aggregate_signal(models, frames[name], year, scheme)
            candidate, active = make_candidate(frames[name], parents[name], signal)
            payload[scheme][name] = signal, candidate, active
            metrics = axis_metrics(frames[name], parents[name], candidate, active)
            metrics["full_row_rms_shift"] = full_row_rms(parents[name], candidate)
            record[name] = metrics
        record["minimum_source_gain"] = min(record["full_2022"]["gain"], record["late_2023"]["gain"])
        record["source_pass"] = bool(
            record["full_2022"]["gain"] > 0.0
            and record["late_2023"]["gain"] > 0.0
            and record["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
            and record["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and record["full_2022"]["worst_month_gain"] > -10.0
            and record["late_2023"]["worst_month_gain"] > -10.0
        )
        grid.append(record)

    passing = [row for row in grid if row["source_pass"]]
    passing.sort(key=lambda row: (row["minimum_source_gain"], row["full_2022"]["gain"], row["late_2023"]["gain"]), reverse=True)
    restrictions = {
        "official_train_only": True,
        "source_residuals_forward_oof": True,
        "strictly_prior_completed_seasons": True,
        "aggregation_selected_before_full_2024": True,
        "fixed_graph_strength_route_and_dose": True,
        "v345_f_and_team13_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "fixed_recipe": {"graph_strength": GRAPH_STRENGTH, "schemes": SCHEMES, "route": "R_CORE", "weight": 0.5},
            "source_screen": grid,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        return summary

    scheme = passing[0]["scheme"]
    signal = aggregate_signal(models, frames["full_2024"], 2024, scheme)
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
    robustness = _robustness(axes24, parents["full_2024"], candidate, active, [parents["full_2024"], candidate])
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in parents},
        candidate_full_2022=payload[scheme]["full_2022"][1],
        candidate_late_2023=payload[scheme]["late_2023"][1],
        candidate_full_2024=candidate,
        active_full_2024=active,
        direction_full_2024=signal,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "fixed_recipe": {"graph_strength": GRAPH_STRENGTH, "selected_scheme": scheme, "route": "R_CORE", "weight": 0.5},
        "source_screen": grid,
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.oof_dir, args.v335_axes, args.v345_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
