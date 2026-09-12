import numpy as np
import pandas as pd

from src.archive.v241_mechanism_aware_fallback_expansion import (
    apply_candidate,
    route_masks,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_type": ["R"] * 7,
            "pitcher_team_id": [1, 1, 1, 1, 1, 13, 1],
            "batter_team_id": [2] * 7,
            "num_runners_on": [1, 1, 0, 0, 0, 1, 0],
            "li": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            "pitcher_hand": ["R", "R", "R", "R", "L", "R", "L"],
            "batter_hand": ["L", "L", "R", "L", "R", "R", "L"],
        }
    )


def test_routes_are_disjoint_and_respect_mechanisms():
    parent = np.array([0.55, 0.49, 0.40, 0.53, 0.51, 0.60, 0.60])
    xgb = np.array([0.60, 0.50, 0.50, 0.60, 0.60, 0.70, np.nan])
    routes = route_masks(parent, xgb, _frame())

    assert np.flatnonzero(routes["deployed"]).tolist() == [0]
    assert np.flatnonzero(routes["pressure_boundary_agreement"]).tolist() == [1]
    assert np.flatnonzero(routes["nonpressure_same_hand"]).tolist() == [2]
    assert np.flatnonzero(routes["nonpressure_opposite_hand_high52"]).tolist() == [3]

    stacked = np.column_stack(list(routes.values())).sum(axis=1)
    assert np.all(stacked <= 1)


def test_candidate_uses_original_parent_and_fixed_route_weights():
    parent = np.array([0.55, 0.49, 0.40, 0.53, 0.51, 0.60, 0.60])
    xgb = np.array([0.60, 0.50, 0.50, 0.60, 0.60, 0.70, np.nan])
    routes = route_masks(parent, xgb, _frame())
    baseline, candidate, changed = apply_candidate(parent, xgb, routes)

    expected_baseline = parent.copy()
    expected_baseline[0] = 0.55 + 0.30 * (0.60 - 0.55)
    expected = expected_baseline.copy()
    expected[1] = 0.49 + 0.30 * (0.50 - 0.49)
    expected[2] = 0.40 + 0.10 * (0.50 - 0.40)
    expected[3] = 0.53 + 0.10 * (0.60 - 0.53)

    np.testing.assert_allclose(baseline, expected_baseline)
    np.testing.assert_allclose(candidate, expected)
    assert np.flatnonzero(changed).tolist() == [1, 2, 3]
