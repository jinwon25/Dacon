"""Exact-axis gain metrics and cluster/reality-check robustness.

Moved verbatim from ``src/archive/v103_fixed_union_robust.py`` during the core extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _diagnostics
from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    grouped_gain_table,
    one_way_cluster_bootstrap,
    white_reality_check,
)


def _axis_metrics(exact: dict[str, np.ndarray], candidate: np.ndarray) -> dict[str, Any]:
    mask = exact["exact_mask"].astype(bool)
    parent = exact["parent"][mask].astype(np.float64)
    axis = {
        "target": exact["target"][mask].astype(np.float64),
        "parent": parent,
        "direct": candidate[mask].astype(np.float64),
        "domain3": exact["domain3"][mask],
        "game_month": exact["game_month"][mask],
    }
    return _diagnostics(axis, ("R_CORE",), 1.0)


def _robust_axis(
    frame: pd.DataFrame,
    exact: dict[str, np.ndarray],
    candidate: np.ndarray,
    family: list[np.ndarray],
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    mask = exact["exact_mask"].astype(bool)
    selected = frame.loc[mask].reset_index(drop=True)
    if "row_id" in exact and not np.array_equal(
        selected["row_id"].astype(str).to_numpy(), exact["row_id"][mask].astype(str)
    ):
        raise ValueError("train/exact row alignment mismatch")
    target = exact["target"][mask].astype(np.float64)
    parent = exact["parent"][mask].astype(np.float64)
    trial = candidate[mask].astype(np.float64)
    repeats = int(config["bootstrap_resamples"])
    seed = int(config["seed"]) + seed_offset
    improvement = np.column_stack(
        [
            np.square(parent - target) - np.square(item[mask] - target)
            for item in family
        ]
    )
    count_state = (
        selected["balls_before"].astype(str)
        + "-"
        + selected["strikes_before"].astype(str)
    )
    pressure = np.where(
        selected["balls_before"].eq(3) | selected["strikes_before"].eq(2),
        "leverage_count",
        "neutral_count",
    )
    pitcher_history = pd.cut(
        selected["asof_pitcher_n"],
        bins=[-np.inf, 29, 199, 999, np.inf],
        labels=["0-29", "30-199", "200-999", "1000+"],
    ).astype(str)
    subgroup_values = {
        "pitcher_hand": selected["pitcher_hand"].astype(str),
        "batter_hand": selected["batter_hand"].astype(str),
        "platoon": (
            selected["pitcher_hand"].astype(str)
            + "-"
            + selected["batter_hand"].astype(str)
        ),
        "count_state": count_state,
        "count_pressure": pressure,
        "pitcher_history": pitcher_history,
    }
    return {
        "pitcher": one_way_cluster_bootstrap(
            target, trial, parent, selected["pitcher_id"],
            n_resamples=repeats, seed=seed,
        ),
        "crossed_pitcher_batter": crossed_pigeonhole_bootstrap(
            target, trial, parent, selected["pitcher_id"], selected["batter_id"],
            n_resamples=repeats, seed=seed + 1,
        ),
        "chronological_block": circular_block_bootstrap(
            target, trial, parent, block_size=int(config["block_size"]),
            n_resamples=repeats, seed=seed + 2,
        ),
        "reality_check": white_reality_check(
            target, improvement, block_size=int(config["block_size"]),
            n_resamples=repeats, seed=seed + 3,
        ),
        "subgroups": {
            key: grouped_gain_table(target, trial, parent, values).to_dict(
                orient="records"
            )
            for key, values in subgroup_values.items()
        },
    }


axis_metrics = _axis_metrics
robust_axis = _robust_axis
