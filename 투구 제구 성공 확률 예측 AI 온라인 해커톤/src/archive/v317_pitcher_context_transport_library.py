"""Bounded strict-forward library of pitcher-by-baseball-context interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.archive.v315_pitcher_calendar_partial_pooling import expit, logit


PROTOCOL = "V317_PITCHER_CONTEXT_TRANSPORT_LIBRARY_V1"
TARGET = "control_success"
PITCHER_SHRINK = 500.0
CONTEXT_SHRINK = 2000.0
CORRECTION_CAP = 0.04
SIGNALS = (
    "count", "inning_phase", "score_bucket", "runner_count",
    "leverage_bucket", "outs", "count_game_type",
)


def add_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    balls = pd.to_numeric(output["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(output["strikes_before"], errors="coerce").fillna(-1).astype(int)
    inning = pd.to_numeric(output["inning"], errors="coerce")
    score = pd.to_numeric(output["score_diff_pitcher_team"], errors="coerce")
    runners = pd.to_numeric(output["num_runners_on"], errors="coerce")
    leverage = pd.to_numeric(output["li"], errors="coerce")
    output["count"] = (balls * 3 + strikes).astype(str)
    output["inning_phase"] = pd.cut(
        inning, [-np.inf, 3, 6, np.inf], labels=["early", "middle", "late"]
    ).astype(str)
    output["score_bucket"] = pd.cut(
        score, [-np.inf, -2, -1, 0, 1, 2, np.inf],
        labels=["trail3", "trail2", "trail1", "tie", "lead1", "lead2plus"],
    ).astype(str)
    output["runner_count"] = runners.clip(0, 3).fillna(-1).astype(int).astype(str)
    output["leverage_bucket"] = pd.cut(
        leverage, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "medium", "high"]
    ).astype(str)
    output["outs"] = pd.to_numeric(output["outs_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    output["count_game_type"] = output["game_type"].astype(str) + "|" + output["count"]
    return output


def fit_lookup(history: pd.DataFrame, context: str) -> tuple[dict[tuple[int, str], float], dict[str, float]]:
    frame = add_contexts(history).dropna(subset=["pitcher_id", TARGET])
    global_rate = float(frame[TARGET].mean())
    pitcher = frame.groupby("pitcher_id", observed=True)[TARGET].agg(["size", "sum"])
    pitcher_rate = (pitcher["sum"] + PITCHER_SHRINK * global_rate) / (
        pitcher["size"] + PITCHER_SHRINK
    )
    context_table = frame.groupby(context, observed=True)[TARGET].agg(["size", "sum"])
    context_rate = (context_table["sum"] + CONTEXT_SHRINK * global_rate) / (
        context_table["size"] + CONTEXT_SHRINK
    )
    cells = frame.groupby(["pitcher_id", context], observed=True)[TARGET].agg(
        ["size", "sum"]
    ).reset_index()
    count = cells["size"].to_numpy(np.float64)
    cell_rate = (cells["sum"].to_numpy(np.float64) + 0.5) / (count + 1.0)
    expected = (
        logit(cells["pitcher_id"].map(pitcher_rate).fillna(global_rate).to_numpy())
        + logit(cells[context].map(context_rate).fillna(global_rate).to_numpy())
        - float(logit(np.asarray([global_rate]))[0])
    )
    raw_delta = logit(cell_rate) - expected
    observed_var = 1.0 / np.clip(count * cell_rate * (1.0 - cell_rate), 1e-6, None)
    stable = count >= 50.0
    tau2 = max(
        float(np.var(raw_delta[stable], ddof=1) - np.mean(observed_var[stable])), 1e-6
    ) if stable.sum() >= 2 else 1e-6
    shrunk = raw_delta * tau2 / (tau2 + observed_var)
    lookup = {
        (int(pitcher), str(value)): float(delta)
        for pitcher, value, delta in zip(cells["pitcher_id"], cells[context], shrunk)
    }
    return lookup, {
        "cells": int(len(cells)), "stable_cells": int(stable.sum()), "tau2": tau2,
    }


def map_direction(
    query: pd.DataFrame,
    parent: np.ndarray,
    context: str,
    lookup: dict[tuple[int, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    frame = add_contexts(query)
    keys = zip(frame["pitcher_id"].astype(int), frame[context].astype(str))
    delta = np.asarray([lookup.get(key, 0.0) for key in keys], dtype=np.float64)
    active = delta != 0.0
    direction = expit(logit(parent) + np.clip(delta, -0.75, 0.75)) - parent
    direction = np.clip(direction, -CORRECTION_CAP, CORRECTION_CAP)
    direction[~active] = 0.0
    return direction, active


def analytic_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def run(train_csv: Path, v285_axes: Path, v290_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "balls_before",
        "strikes_before", "inning", "score_diff_pitcher_team", "num_runners_on",
        "li", "outs_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    target = {name: frame[TARGET].to_numpy(np.float64) for name, frame in frames.items()}
    histories = {
        "full_2022": train.loc[train["season"].between(2019, 2021)].reset_index(drop=True),
        "late_2023": train.loc[train["season"].between(2020, 2022)].reset_index(drop=True),
        "full_2024": train.loc[train["season"].between(2021, 2023)].reset_index(drop=True),
    }
    source_trials: list[dict[str, Any]] = []
    source_banks: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    diagnostics: dict[str, Any] = {}
    for signal in SIGNALS:
        source_banks[signal] = {}
        diagnostics[signal] = {}
        for name in ("full_2022", "late_2023"):
            lookup, diag = fit_lookup(histories[name], signal)
            direction, active = map_direction(frames[name], parent[name], signal, lookup)
            source_banks[signal][name] = (direction, active)
            diagnostics[signal][name] = {**diag, "active_rate": float(active.mean())}
        dose = analytic_dose([
            (target[name][source_banks[signal][name][1]] - parent[name][source_banks[signal][name][1]],
             source_banks[signal][name][0][source_banks[signal][name][1]])
            for name in ("full_2022", "late_2023")
        ])
        per_axis = {}
        for name in ("full_2022", "late_2023"):
            direction, active = source_banks[signal][name]
            candidate = np.clip(parent[name] + dose * direction, 0.001, 0.999)
            per_axis[name] = metrics(frames[name], target[name], parent[name], candidate, active)
        passes = bool(
            dose > 0.0
            and all(per_axis[name]["gain"] > 0.0 for name in per_axis)
            and per_axis["full_2022"]["positive_month_fraction"] >= 5.0 / 7.0
            and per_axis["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and min(per_axis[name]["worst_month_gain"] for name in per_axis) > -3.0
        )
        source_trials.append({
            "signal": signal, "dose": dose, "passes": passes,
            "minimum_gain": min(per_axis[name]["gain"] for name in per_axis),
            "mean_gain": float(np.mean([per_axis[name]["gain"] for name in per_axis])),
            "metrics": per_axis,
        })
    eligible = [trial for trial in source_trials if trial["passes"]]
    pool = eligible if eligible else source_trials
    selected = max(pool, key=lambda item: (item["passes"], item["minimum_gain"], item["mean_gain"]))
    signal = str(selected["signal"])
    dose = float(selected["dose"])
    lookup24, diag24 = fit_lookup(histories["full_2024"], signal)
    direction24, active24 = map_direction(
        frames["full_2024"], parent["full_2024"], signal, lookup24
    )
    candidate24 = np.clip(parent["full_2024"] + dose * direction24, 0.001, 0.999)
    locked = metrics(
        frames["full_2024"], target["full_2024"], parent["full_2024"],
        candidate24, active24,
    )
    locked_pass = bool(
        selected["passes"]
        and locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent["full_2024"], candidate_full_2024=candidate24,
        direction_full_2024=direction24, active_full_2024=active24,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if locked_pass else (
            "locked_reject" if selected["passes"] else "source_reject"
        ),
        "candidate_count": len(SIGNALS),
        "selected": selected,
        "source_trials": source_trials,
        "locked_full_2024": locked,
        "locked_diagnostics": {**diag24, "active_rate": float(active24.mean())},
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "candidate_library_predeclared": True,
            "selection_uses_only_2022_and_late_2023": True,
            "only_selected_signal_opened_on_full_2024": True,
            "evaluation_row_order_or_aggregates_used": False,
            "runtime_live_features_are_row_local": True,
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
