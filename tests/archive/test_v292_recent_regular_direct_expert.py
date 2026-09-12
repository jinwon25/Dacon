from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v292_recent_regular_direct_expert import WEIGHT, _blend, _routes


def test_routes_partition_regular_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F", "R"],
            "pitcher_team_id": [1, 13, 1, 2],
            "batter_team_id": [2, 2, 13, 13],
        }
    )
    routes = _routes(frame)
    np.testing.assert_array_equal(routes["R_ALL"], [True, True, False, True])
    np.testing.assert_array_equal(routes["R_CORE"], [True, False, False, False])
    np.testing.assert_array_equal(routes["R_ANCHOR"], [False, True, False, True])
    np.testing.assert_array_equal(
        routes["R_CORE"] | routes["R_ANCHOR"], routes["R_ALL"]
    )


def test_blend_preserves_inactive_rows_and_fixed_weight() -> None:
    parent = np.array([0.2, 0.4, 0.8])
    expert = np.array([0.4, 0.1, 0.6])
    active = np.array([True, False, True])
    actual = _blend(parent, expert, active)
    expected = parent.copy()
    expected[active] += WEIGHT * (expert[active] - parent[active])
    np.testing.assert_allclose(actual, expected)
