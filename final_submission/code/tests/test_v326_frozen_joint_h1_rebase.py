import numpy as np
import pandas as pd

from src.archive.v326_frozen_joint_h1_rebase import r_core


def test_r_core_protects_f_and_anchor_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F", "R"],
            "pitcher_team_id": [12, 13, 12, 12],
            "batter_team_id": [14, 14, 14, 13],
        }
    )
    assert np.array_equal(r_core(frame), np.asarray([True, False, False, False]))
