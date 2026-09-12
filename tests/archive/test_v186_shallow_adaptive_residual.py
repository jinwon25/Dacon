import numpy as np
import pandas as pd

from src.archive.v186_shallow_adaptive_residual import apply_correction, meta_features


def test_apply_correction_is_row_local_and_active_only() -> None:
    base = np.array([0.4, 0.5, 0.6])
    correction = np.array([0.01, -0.02, 0.03])
    active = np.array([True, False, True])
    result = apply_correction(base, correction, active, 0.5)
    np.testing.assert_allclose(result, [0.405, 0.5, 0.615])


def test_meta_features_exclude_player_identity() -> None:
    required = {
        "top_bottom": ["T"], "game_type": ["R"], "base_state": ["000"],
        "pitcher_hand": [1], "batter_hand": [2],
    }
    for column in (
        "game_month", "inning", "balls_before", "strikes_before", "outs_before",
        "run_total_before", "score_diff_pitcher_team", "num_runners_on",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "home_win_expectancy",
        "away_win_expectancy", "li", "asof_pitcher_n", "asof_batter_n",
        "asof_pitcher_pitchmix_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ):
        required[column] = [0.5]
    frame = pd.DataFrame(required)
    features, names, _ = meta_features(frame, np.array([0.5]), np.array([0.51]))
    assert "pitcher_id" not in names
    assert "batter_id" not in names
    assert features.shape[0] == 1
