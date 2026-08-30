import numpy as np
import pandas as pd

from src.archive.v251_anchor_dual_tree_specialist import (
    FAMILIES,
    WEIGHTS,
    anchor_mask,
    apply_specialist,
    restrictions,
)


def test_anchor_mask_is_regular_team13_only() -> None:
    frame = pd.DataFrame({
        "game_type": ["R", "R", "F"],
        "pitcher_team_id": [13, 1, 13],
        "batter_team_id": [2, 13, 2],
    })
    assert anchor_mask(frame).tolist() == [True, True, False]


def test_specialist_preserves_inactive_rows() -> None:
    parent = np.array([0.4, 0.6])
    specialist = np.array([0.8, 0.2])
    output = apply_specialist(parent, specialist, np.array([True, False]), 0.1)
    np.testing.assert_allclose(output, [0.44, 0.6])


def test_contract_is_low_complexity_and_public_blind() -> None:
    assert len(FAMILIES) == 3
    assert max(WEIGHTS) <= 0.30
    audit = restrictions()
    assert audit["new_route_is_disjoint_r_anchor_only"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
