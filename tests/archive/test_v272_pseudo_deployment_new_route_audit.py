import numpy as np
import pandas as pd

from src.archive.v272_pseudo_deployment_new_route_audit import (
    apply_new_route,
    new_route_masks,
)


def test_new_route_masks_split_untouched_core_and_anchor_in_calendar() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "R", "R"],
            "pitcher_team_id": [1, 1, 13, 1],
            "batter_team_id": [2, 2, 2, 2],
            "game_month": [5, 5, 5, 3],
        }
    )
    masks = new_route_masks(
        frame,
        np.array([True, False, False, False]),
        np.array([0.4, 0.4, 0.4, 0.4]),
    )
    np.testing.assert_array_equal(masks["core_complement"], [False, True, False, False])
    np.testing.assert_array_equal(masks["anchor"], [False, False, True, False])
    np.testing.assert_array_equal(masks["core_plus_anchor"], [False, True, True, False])


def test_apply_new_route_moves_from_parent_at_fixed_low_dose() -> None:
    candidate = apply_new_route(
        np.array([0.5, 0.5]),
        np.array([0.4, 0.4]),
        np.array([0.6, 0.6]),
        np.array([True, False]),
    )
    np.testing.assert_allclose(candidate, [0.43, 0.5])
