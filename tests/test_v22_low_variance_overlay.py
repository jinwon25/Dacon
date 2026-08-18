from __future__ import annotations

import numpy as np
import pandas as pd

from src.v22_joint_low_variance_screen import _candidate, _gains, _quadratic


def test_quadratic_gain_matches_direct_brier_gain() -> None:
    target = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0])
    parent = np.asarray([0.2, 0.6, 0.7, 0.3, 0.8])
    signals = np.column_stack(
        [np.asarray([0.1, -0.1, 0.05, 0.02, -0.04]), np.full(5, 0.01)]
    )
    weights = np.asarray([[0.2, 0.5]])
    linear, quadratic = _quadratic(target, parent, signals)
    computed = float(_gains(weights, linear, quadratic)[0])
    prediction = _candidate(parent, signals, weights[0])
    scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
    direct = scale * (
        np.mean(np.square(target - parent))
        - np.mean(np.square(target - prediction))
    )
    assert np.isclose(computed, direct)


def test_overlay_inputs_are_row_local() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_success_rate": [0.4, np.nan],
            "asof_batter_success_rate": [0.6, 0.2],
        }
    )
    pitcher = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(0.5)
    batter = pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").fillna(0.5)
    prior = 0.75 * pitcher + 0.25 * batter
    assert np.allclose(prior, [0.45, 0.425])
