from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v349_route_residual_booster_v345 import apply_routes, route_masks


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "game_type": ["F", "R", "R", "R"],
        "pitcher_team_id": [1, 13, 1, 1],
        "batter_team_id": [2, 2, 13, 2],
    })


def test_route_masks_partition_fixture() -> None:
    masks = route_masks(_frame())
    assert [int(masks[name].sum()) for name in ("F", "R_ANCHOR", "R_CORE")] == [1, 2, 1]
    assert np.all(sum(masks.values()) == 1)


def test_apply_routes_protects_unretained_routes() -> None:
    frame = _frame()
    parent = np.array([0.2, 0.3, 0.4, 0.5])
    corrections = {
        "F": np.array([0.1]),
        "R_ANCHOR": np.array([-0.1, -0.2]),
        "R_CORE": np.array([0.2]),
    }
    candidate, active = apply_routes(frame, parent, corrections, {"R_CORE"})
    np.testing.assert_allclose(candidate, np.array([0.2, 0.3, 0.4, 0.7]))
    np.testing.assert_array_equal(active, np.array([False, False, False, True]))


def test_apply_routes_allows_empty_retained_set() -> None:
    frame = _frame()
    parent = np.array([0.2, 0.3, 0.4, 0.5])
    corrections = {
        "F": np.array([0.1]),
        "R_ANCHOR": np.array([-0.1, -0.2]),
        "R_CORE": np.array([0.2]),
    }
    candidate, active = apply_routes(frame, parent, corrections, set())
    np.testing.assert_allclose(candidate, parent)
    assert not np.any(active)
