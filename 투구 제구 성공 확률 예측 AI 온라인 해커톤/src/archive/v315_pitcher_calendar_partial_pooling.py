"""Strict-forward pitcher-by-calendar-month partial pooling above v290.

The hypothesis is a repeatable pitcher-specific seasonal trajectory (ramp-up,
summer fatigue, rehabilitation or role cadence).  A hierarchical logit table
removes pitcher and calendar-month main effects, estimates its interaction
variance from prior seasons only, and maps the shrunk interaction row-locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import metrics


PROTOCOL = "V315_PITCHER_CALENDAR_PARTIAL_POOLING_V1"
TARGET = "control_success"
PITCHER_SHRINK = 500.0
MONTH_SHRINK = 2000.0
CORRECTION_CAP = 0.04


def logit(values: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def fit_calendar_lookup(history: pd.DataFrame) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    frame = history.dropna(subset=["pitcher_id", "game_month", TARGET]).copy()
    global_rate = float(frame[TARGET].mean())
    pitcher = frame.groupby("pitcher_id", observed=True)[TARGET].agg(["size", "sum"])
    pitcher_rate = (pitcher["sum"] + PITCHER_SHRINK * global_rate) / (
        pitcher["size"] + PITCHER_SHRINK
    )
    month = frame.groupby("game_month", observed=True)[TARGET].agg(["size", "sum"])
    month_rate = (month["sum"] + MONTH_SHRINK * global_rate) / (
        month["size"] + MONTH_SHRINK
    )
    cells = frame.groupby(["pitcher_id", "game_month"], observed=True)[TARGET].agg(
        ["size", "sum"]
    ).reset_index()
    cell_rate = (
        cells["sum"].to_numpy(np.float64) + 0.5
    ) / (cells["size"].to_numpy(np.float64) + 1.0)
    expected_logit = (
        logit(cells["pitcher_id"].map(pitcher_rate).fillna(global_rate).to_numpy())
        + logit(cells["game_month"].map(month_rate).fillna(global_rate).to_numpy())
        - float(logit(np.asarray([global_rate]))[0])
    )
    raw_delta = logit(cell_rate) - expected_logit
    observed_var = 1.0 / np.clip(
        cells["size"].to_numpy(np.float64) * cell_rate * (1.0 - cell_rate),
        1e-6,
        None,
    )
    stable = cells["size"].to_numpy(np.float64) >= 80.0
    tau2 = max(
        float(np.var(raw_delta[stable], ddof=1) - np.mean(observed_var[stable])),
        1e-6,
    ) if stable.sum() >= 2 else 1e-6
    shrunk = raw_delta * tau2 / (tau2 + observed_var)
    lookup = {
        (int(pitcher_id), int(game_month)): float(delta)
        for pitcher_id, game_month, delta in zip(
            cells["pitcher_id"], cells["game_month"], shrunk
        )
    }
    return lookup, {
        "global_rate": global_rate,
        "cells": int(len(cells)),
        "stable_cells": int(stable.sum()),
        "tau2": tau2,
    }


def map_calendar_direction(
    query: pd.DataFrame,
    parent: np.ndarray,
    lookup: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    keys = zip(query["pitcher_id"].astype(int), query["game_month"].astype(int))
    delta = np.asarray([lookup.get(key, 0.0) for key in keys], dtype=np.float64)
    active = delta != 0.0
    direction = expit(logit(parent) + np.clip(delta, -0.75, 0.75)) - parent
    direction = np.clip(direction, -CORRECTION_CAP, CORRECTION_CAP)
    direction[~active] = 0.0
    return direction, active


def pooled_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def run(train_csv: Path, v285_axes: Path, v290_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", TARGET],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parents.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    year_by_axis = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    directions: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, year in year_by_axis.items():
        history = train.loc[
            train["season"].lt(year) & train["season"].ge(year - 3)
        ].reset_index(drop=True)
        lookup, diagnostics[name] = fit_calendar_lookup(history)
        directions[name], active[name] = map_calendar_direction(
            frames[name], parents[name], lookup
        )
        diagnostics[name]["active_rate"] = float(active[name].mean())
    target = {name: frame[TARGET].to_numpy(np.float64) for name, frame in frames.items()}
    dose = pooled_dose([
        (target["full_2022"][active["full_2022"]] - parents["full_2022"][active["full_2022"]],
         directions["full_2022"][active["full_2022"]]),
        (target["late_2023"][active["late_2023"]] - parents["late_2023"][active["late_2023"]],
         directions["late_2023"][active["late_2023"]]),
    ])
    candidates = {
        name: np.clip(parents[name] + dose * directions[name], 0.001, 0.999)
        for name in frames
    }
    results = {
        name: metrics(frame, target[name], parents[name], candidates[name], active[name])
        for name, frame in frames.items()
    }
    source_pass = bool(
        dose > 0.0
        and results["full_2022"]["gain"] > 0.0
        and results["late_2023"]["gain"] > 0.0
        and results["full_2022"]["positive_month_fraction"] >= 0.75
        and results["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
    )
    locked_pass = bool(
        results["full_2024"]["gain"] > 0.0
        and results["full_2024"]["positive_month_fraction"] >= 0.625
        and results["full_2024"]["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
        **{f"active_{name}": active[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "source_selected_dose": dose,
        "metrics": results,
        "diagnostics": diagnostics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_three_seasons": True,
            "evaluation_row_order_or_aggregates_used": False,
            "runtime_live_features_are_row_local": True,
            "full_2024_used_for_dose_selection": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v285_axes, args.v290_axes, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
