"""Persistent cross-season count-graph correction above v345.

This is the stability-filtered continuation of v365.  The graph recipe is
frozen at the source-ranked strength 300 without opening 2024.  A row is
corrected only when at least two strictly prior completed-season pitcher by
count-hand graph estimates are available and every available estimate has the
same sign.  The correction is their mean and is applied only on R_CORE.
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
from src.archive.v365_count_graph_partial_pooling_v345 import fit_model, map_model, make_candidate


PROTOCOL = "V366_PERSISTENT_COUNT_GRAPH_V345_V1"
TARGET = "control_success"
GRAPH_STRENGTH = 300.0


def persistent_signal(models: dict[int, dict[str, Any]], rows: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.vstack([
        map_model(models[source_year], rows)
        for source_year in SOURCE_YEARS if source_year < year
    ])
    available = np.abs(mapped) > 1e-15
    support = available.sum(axis=0)
    positive = ((mapped > 0.0) & available).sum(axis=0)
    negative = ((mapped < 0.0) & available).sum(axis=0)
    stable = (support >= 2) & ((positive == support) | (negative == support))
    signal = np.zeros(len(rows), dtype=np.float64)
    signal[stable] = mapped[:, stable].sum(axis=0) / support[stable]
    return signal, support


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

    signals: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, year in (("full_2022", 2022), ("late_2023", 2023)):
        signals[name], supports[name] = persistent_signal(models, frames[name], year)
        candidates[name], active[name] = make_candidate(frames[name], parents[name], signals[name])
        metrics[name] = axis_metrics(frames[name], parents[name], candidates[name], active[name])
        metrics[name]["full_row_rms_shift"] = full_row_rms(parents[name], candidates[name])
        metrics[name]["median_prior_support_active"] = float(np.median(supports[name][active[name]])) if active[name].any() else 0.0

    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0
        and metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
        and metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
        and metrics["full_2022"]["worst_month_gain"] > -10.0
        and metrics["late_2023"]["worst_month_gain"] > -10.0
    )
    restrictions = {
        "official_train_only": True,
        "source_residuals_forward_oof": True,
        "strictly_prior_completed_seasons": True,
        "graph_recipe_frozen_from_v365_source_only_screen": True,
        "full_2024_opened_only_after_source_gate": True,
        "v345_f_and_team13_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not source_pass:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "fixed_recipe": {"graph_strength": GRAPH_STRENGTH, "minimum_prior_seasons": 2, "unanimous_available_sign": True, "route": "R_CORE", "weight": 0.5},
            "source_metrics": metrics,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        return summary

    signals["full_2024"], supports["full_2024"] = persistent_signal(models, frames["full_2024"], 2024)
    candidates["full_2024"], active["full_2024"] = make_candidate(
        frames["full_2024"], parents["full_2024"], signals["full_2024"]
    )
    locked = axis_metrics(frames["full_2024"], parents["full_2024"], candidates["full_2024"], active["full_2024"])
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], candidates["full_2024"])
    locked["median_prior_support_active"] = float(np.median(supports["full_2024"][active["full_2024"]])) if active["full_2024"].any() else 0.0
    delta = candidates["full_2024"] - parents["full_2024"]
    union = (np.abs(delta) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(delta[union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(axes24, parents["full_2024"], candidates["full_2024"], active["full_2024"], [parents["full_2024"], candidates["full_2024"]])
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in parents},
        **{f"candidate_{name}": candidates[name] for name in candidates},
        active_full_2024=active["full_2024"],
        direction_full_2024=signals["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "fixed_recipe": {"graph_strength": GRAPH_STRENGTH, "minimum_prior_seasons": 2, "unanimous_available_sign": True, "route": "R_CORE", "weight": 0.5},
        "source_metrics": metrics,
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
