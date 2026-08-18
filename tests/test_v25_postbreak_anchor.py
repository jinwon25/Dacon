from __future__ import annotations

import numpy as np
import pandas as pd

from src.v10_overlay_script import _v25_postbreak_frame
from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_structural_residual_screen import _derived


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "balls_before": [0, 3],
            "strikes_before": [0, 2],
            "pitcher_hand": [1, 2],
            "batter_hand": [2, 1],
            "inning": [2, 8],
            "game_type": ["R", "F"],
            "pitcher_team_id": [13, 2],
            "batter_team_id": [4, 3],
            "asof_pitcher_n": [0, 100],
            "asof_batter_n": [10, 20],
            "asof_pitcher_pitchmix_n": [5, 7],
            "asof_pitcher_prev1_game_success_rate": [0.4, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.3, 0.6],
            "asof_pitcher_prev1_game_middle_rate": [0.2, 0.7],
            "asof_pitcher_prev5_game_middle_rate": [0.1, 0.5],
            "asof_pitcher_success_rate": [0.55, 0.45],
            "asof_batter_success_rate": [0.50, 0.40],
        }
    )


def test_packaged_feature_derivation_matches_training() -> None:
    frame = _frame()
    expected = _derived(frame, _joint_domain(frame))
    actual = _v25_postbreak_frame(frame)
    columns = [
        "domain3",
        "count_state",
        "hand_matchup",
        "inning_bucket",
        "log1p_asof_pitcher_n",
        "log1p_asof_batter_n",
        "log1p_asof_pitcher_pitchmix_n",
        "recent_success_delta_1_5",
        "recent_middle_delta_1_5",
        "pitcher_batter_rate_gap",
    ]
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_joint_domain_marks_team_13_regular_as_anchor() -> None:
    assert _joint_domain(_frame()).tolist() == ["R_ANCHOR", "F"]
    assert np.isfinite(_v25_postbreak_frame(_frame())["pitcher_batter_rate_gap"]).all()
