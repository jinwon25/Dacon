"""Strict recent-regime F low-rank interaction audit above v290.

This is deliberately not another blend-weight search.  A single, predeclared
rank-2 empirical-Bayes matrix learns the residual interaction between pitcher
and the 24 (count, batter-hand) states.  The residual is first stripped of
pitcher and context main effects, so the direction is complementary to v290's
direct recent-F expert.

Early-2023 F rows fit the source matrix and late-2023 F rows select one bounded
analytic dose.  The dose and matrix recipe are then frozen; all 2023 F rows
fit the transfer matrix and full-2024 is opened once as the locked axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import bss, metrics
from src.champion.v50_low_rank_pitcher_context import (
    fit_source_matrix,
    map_source_matrix,
)


PROTOCOL = "V313_RECENT_F_LOWRANK_INTERACTION_V1"
TARGET = "control_success"
SMOOTHING = 300.0
RANK = 2
PITCHER_MAIN_SHRINK = 500.0
CONTEXT_MAIN_SHRINK = 1000.0
MAX_DOSE = 1.0
CORRECTION_CAP = 0.025


def additive_source_parent(rows: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Return a source-only additive main-effect parent for interaction fitting."""

    frame = rows[["pitcher_id", "balls_before", "strikes_before", "batter_hand"]].copy()
    y = np.asarray(target, dtype=np.float64)
    if len(frame) != len(y):
        raise ValueError("source row/target length mismatch")
    global_rate = float(y.mean())
    pitcher = pd.DataFrame({"pitcher_id": frame["pitcher_id"], "y": y}).groupby(
        "pitcher_id", observed=True
    )["y"].agg(["size", "sum"])
    pitcher_rate = (pitcher["sum"] + PITCHER_MAIN_SHRINK * global_rate) / (
        pitcher["size"] + PITCHER_MAIN_SHRINK
    )
    context_code = (
        (pd.to_numeric(frame["balls_before"]) * 3 + pd.to_numeric(frame["strikes_before"]))
        * 2
        + (pd.to_numeric(frame["batter_hand"]) - 1)
    ).astype(np.int16)
    context = pd.DataFrame({"context": context_code, "y": y}).groupby(
        "context", observed=True
    )["y"].agg(["size", "sum"])
    context_rate = (context["sum"] + CONTEXT_MAIN_SHRINK * global_rate) / (
        context["size"] + CONTEXT_MAIN_SHRINK
    )
    output = (
        frame["pitcher_id"].map(pitcher_rate).fillna(global_rate).to_numpy(np.float64)
        + context_code.map(context_rate).fillna(global_rate).to_numpy(np.float64)
        - global_rate
    )
    return np.clip(output, 0.02, 0.98)


def fit_direction(source: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the frozen matrix recipe and map it row-locally to a future frame."""

    target = source[TARGET].to_numpy(np.float64)
    parent = additive_source_parent(source, target)
    model = fit_source_matrix(
        source,
        target,
        parent,
        smoothing_grid=(SMOOTHING,),
        rank_grid=(RANK,),
    )
    bank, seen, exact_seen = map_source_matrix(model, query)
    key = f"lowrank_s{int(SMOOTHING)}_r{RANK}"
    direction = np.clip(bank[key], -CORRECTION_CAP, CORRECTION_CAP)
    diagnostics = dict(model["diagnostics"])
    diagnostics.update(
        {
            "query_rows": int(len(query)),
            "pitcher_seen_rate": float(seen.mean()),
            "exact_context_seen_rate": float(exact_seen.mean()),
            "mean_abs_direction_seen": float(np.mean(np.abs(direction[seen]))) if seen.any() else 0.0,
        }
    )
    return direction, seen, diagnostics


def analytic_dose(target: np.ndarray, parent: np.ndarray, direction: np.ndarray, active: np.ndarray) -> float:
    residual = np.asarray(target, dtype=np.float64)[active] - np.asarray(parent, dtype=np.float64)[active]
    vector = np.asarray(direction, dtype=np.float64)[active]
    denominator = float(np.dot(vector, vector))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(residual, vector) / denominator, 0.0, MAX_DOSE))


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "recent_regime_fit_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "source_recipe_predeclared": True,
        "source_dose_frozen_before_full_2024": True,
        "public_score_used_for_selection": False,
    }


def run(train_csv: Path, v290_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "balls_before",
        "strikes_before", "batter_hand", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    is_f = train["game_type"].astype(str).eq("F")
    early_f_2023 = train.loc[
        train["season"].eq(2023) & is_f & train["game_month"].lt(8)
    ].reset_index(drop=True)
    all_f_2023 = train.loc[train["season"].eq(2023) & is_f].reset_index(drop=True)
    frames = {
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v290_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in frames
        }
    for name, frame in frames.items():
        if len(frame) != len(parents[name]):
            raise ValueError(f"v290 axis length mismatch: {name}")

    source_direction, source_seen, source_diag = fit_direction(
        early_f_2023, frames["late_2023"]
    )
    transfer_direction, transfer_seen, transfer_diag = fit_direction(
        all_f_2023, frames["full_2024"]
    )
    directions = {"late_2023": source_direction, "full_2024": transfer_direction}
    seen = {"late_2023": source_seen, "full_2024": transfer_seen}
    active = {
        name: frame["game_type"].astype(str).eq("F").to_numpy() & seen[name]
        for name, frame in frames.items()
    }
    directions = {
        name: np.where(active[name], directions[name], 0.0)
        for name in frames
    }
    target = {name: frame[TARGET].to_numpy(np.float64) for name, frame in frames.items()}
    dose = analytic_dose(
        target["late_2023"], parents["late_2023"], directions["late_2023"], active["late_2023"]
    )
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
        and results["late_2023"]["gain"] > 0.0
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
        "recipe": {
            "smoothing": SMOOTHING,
            "rank": RANK,
            "pitcher_main_shrink": PITCHER_MAIN_SHRINK,
            "context_main_shrink": CONTEXT_MAIN_SHRINK,
            "correction_cap": CORRECTION_CAP,
        },
        "metrics": results,
        "diagnostics": {"late_2023": source_diag, "full_2024": transfer_diag},
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v290_axes, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
