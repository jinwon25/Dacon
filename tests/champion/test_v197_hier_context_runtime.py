from __future__ import annotations

import numpy as np
import pandas as pd

from src.champion.v197_finalize_hier_context import build_opening_snapshot
from src.champion.v197_hier_context_runtime import (
    combine_hierarchical_residual,
    hierarchical_features_for_inference,
)


def test_opening_snapshot_and_inference_are_row_local() -> None:
    history = pd.DataFrame(
        {
            "pitcher_id": [1, 2, 1],
            "asof_pitcher_n": [0.0, 0.0, 1.0],
            "asof_pitcher_success_rate": [0.5, 0.5, 1.0],
        }
    )
    target = np.array([1.0, 0.0, 0.0])
    snapshot = build_opening_snapshot(history, target, np.ones(3, dtype=bool))
    assert snapshot["pitcher_opening"][1] == (2.0, 1.0)
    assert snapshot["pitcher_opening"][2] == (1.0, 0.0)

    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 999],
            "asof_pitcher_n": [2.0, 0.0],
            "asof_pitcher_success_rate": [0.5, 0.5],
            "asof_pitcher_prev1_game_success_rate": [0.5, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.5, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.5, np.nan],
        }
    )
    first = hierarchical_features_for_inference(frame.iloc[[0]], snapshot)
    together = hierarchical_features_for_inference(frame, snapshot).iloc[[0]]
    np.testing.assert_allclose(first.to_numpy(), together.to_numpy(), rtol=0, atol=0)
    assert np.isfinite(hierarchical_features_for_inference(frame, snapshot)).all().all()


def test_hierarchy_and_residual_are_composed_before_clipping() -> None:
    output = combine_hierarchical_residual(
        np.array([0.45, 0.99, 0.01]), np.array([0.03, 0.20, -0.20])
    )
    np.testing.assert_allclose(output, [0.48, 0.999, 0.001])
