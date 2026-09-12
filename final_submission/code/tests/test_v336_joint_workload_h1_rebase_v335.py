import numpy as np
import pandas as pd

from src.archive import v336_joint_workload_h1_rebase_v335 as v336


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_type": ["R", "R", "F", "R"],
            "pitcher_team_id": [12, 13, 12, 12],
            "batter_team_id": [14, 14, 14, 13],
            "num_runners_on": [0, 1, 0, 0],
            "li": [0.5, 2.0, 0.4, 0.3],
        }
    )


def test_route_masks_are_disjoint_and_cover_rows() -> None:
    masks = v336.route_masks(_frame())
    assert np.array_equal(masks["R_CORE_ALL"], [True, False, False, False])
    assert np.array_equal(masks["R_ANCHOR"], [False, True, False, True])
    assert np.array_equal(masks["F"], [False, False, True, False])
    assert np.array_equal(
        masks["R_CORE_ALL"] | masks["R_ANCHOR"] | masks["F"],
        np.ones(4, dtype=bool),
    )
    assert not np.any(masks["R_CORE_ALL"] & masks["R_ANCHOR"])


def test_apply_direction_preserves_inactive_rows() -> None:
    parent = np.array([0.2, 0.4, 0.6])
    direction = np.array([0.1, -0.2, 0.3])
    active = np.array([True, False, True])
    candidate = v336.apply_direction(parent, direction, active, 0.25)
    assert np.allclose(candidate, [0.225, 0.4, 0.675])
    assert candidate[1] == parent[1]


def test_full_row_rms_uses_all_rows() -> None:
    parent = np.zeros(4)
    candidate = np.array([0.0, 0.0, 0.0, 2.0])
    assert v336.full_row_rms(parent, candidate) == 1.0


def test_source_selection_cannot_receive_locked_origin() -> None:
    metric = {
        "gain": 1.0,
        "positive_month_fraction": 1.0,
        "worst_month_gain": 0.1,
    }
    screen = {
        scope: {
            "0.100": {
                "full_2022": dict(metric),
                "late_2023": dict(metric),
                "full_2024": dict(metric),
            }
        }
        for scope in v336.PROMOTION_SCOPES
    }
    try:
        v336.select_source_candidate(screen)
    except ValueError as error:
        assert "non-source" in str(error)
    else:
        raise AssertionError("locked origin must not enter source selection")
