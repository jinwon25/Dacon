"""Shared cluster, chronological, and family-wise robustness evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    one_way_cluster_bootstrap,
    white_reality_check,
)


def evaluate_robustness(
    axis: dict[str, np.ndarray],
    candidate: np.ndarray,
    family: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate without assuming independent pitch-level rows."""
    target = axis["target"].astype(np.float64)
    parent = axis["parent"].astype(np.float64)
    repeats = int(config["bootstrap_resamples"])
    seed = int(config["seed"])
    block_size = int(config["block_size"])
    improvement = np.column_stack(
        [np.square(parent - target) - np.square(item - target) for item in family]
    )
    return {
        "pitcher": one_way_cluster_bootstrap(
            target,
            candidate,
            parent,
            axis["pitcher_id"],
            n_resamples=repeats,
            seed=seed,
        ),
        "crossed_pitcher_batter": crossed_pigeonhole_bootstrap(
            target,
            candidate,
            parent,
            axis["pitcher_id"],
            axis["batter_id"],
            n_resamples=repeats,
            seed=seed + 1,
        ),
        "chronological_block": circular_block_bootstrap(
            target,
            candidate,
            parent,
            block_size=block_size,
            n_resamples=repeats,
            seed=seed + 2,
        ),
        "reality_check": white_reality_check(
            target,
            improvement,
            block_size=block_size,
            n_resamples=repeats,
            seed=seed + 3,
        ),
    }
