import numpy as np
import pandas as pd

from src.archive.v120_recent_anchor_spread import (
    apply_spread,
    fit_alpha,
    recent_anchor,
    route_mask,
)


def test_recent_anchor_uses_available_values_and_career_fallback():
    frame = pd.DataFrame({
        "asof_pitcher_prev1_game_success_rate": [0.4, np.nan, np.nan],
        "asof_pitcher_prev3_game_success_rate": [0.6, 0.7, np.nan],
        "asof_pitcher_prev5_game_success_rate": [0.8, np.nan, np.nan],
        "asof_pitcher_success_rate": [0.5, 0.55, np.nan],
    })
    np.testing.assert_allclose(recent_anchor(frame, "prev1"), [0.4, 0.55, 0.5])
    np.testing.assert_allclose(recent_anchor(frame, "mean135"), [0.6, 0.7, 0.5])
    np.testing.assert_allclose(recent_anchor(frame, "weighted135"), [0.54, 0.7, 0.5])


def test_fit_and_apply_spread_are_exact_and_route_local():
    anchor = np.asarray([0.5, 0.5, 0.5])
    parent = np.asarray([0.4, 0.5, 0.6])
    target = anchor + 1.1 * (parent - anchor)
    active = np.asarray([True, True, True])
    alpha = fit_alpha([(target, parent, anchor, active)], 0.85, 1.15)
    assert np.isclose(alpha, 1.1)
    candidate = apply_spread(parent, anchor, np.asarray([True, False, True]), alpha)
    np.testing.assert_allclose(candidate, [0.39, 0.5, 0.61])
    np.testing.assert_array_equal(
        route_mask(np.asarray(["R_CORE", "R_ANCHOR", "F"]), "R_CORE_R_ANCHOR"),
        [True, True, False],
    )
