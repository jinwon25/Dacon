import numpy as np
import pandas as pd

from src.core.axes import _joint_domain
from src.v23_multiyear_direct_screen import _candidate


def test_joint_domain_is_row_local_and_matches_champion_routing():
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 13, 1],
            "batter_team_id": [2, 2, 13],
        }
    )
    assert _joint_domain(frame).tolist() == ["R_CORE", "R_ANCHOR", "F"]


def test_direct_candidate_is_convex_blend():
    parent = np.array([0.2, 0.8])
    direct = np.array([0.6, 0.4])
    np.testing.assert_allclose(_candidate(parent, direct, 0.25), [0.3, 0.7])
