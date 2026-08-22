import numpy as np
import pandas as pd

from src.v97_conditional_direct_forward import _apply, _bank, _feature_frame


def _rows() -> pd.DataFrame:
    columns = {
        "row_id": ["a", "b", "c"], "season": [2020, 2020, 2021],
        "game_month": [4, 5, 4], "game_dayofweek": [1, 2, 3],
        "inning": [1, 2, 3], "top_bottom": ["T", "B", "T"],
        "game_type": ["R", "R", "F"], "balls_before": [0, 3, 0],
        "strikes_before": [0, 2, 1], "outs_before": [0, 1, 2],
        "run_top_before": [0, 0, 1], "run_bot_before": [0, 1, 0],
        "run_total_before": [0, 1, 1], "score_diff_home": [0, 1, -1],
        "score_diff_pitcher_team": [0, -1, 1], "runner_on_1b": [0, 1, 0],
        "runner_on_2b": [0, 0, 1], "runner_on_3b": [0, 0, 0],
        "num_runners_on": [0, 1, 1], "base_state": ["___", "1__", "_2_"],
        "home_win_expectancy": [50.0, 55.0, 45.0], "away_win_expectancy": [50.0, 45.0, 55.0],
        "li": [1.0, 1.5, 0.8], "pitcher_id": [10, 10, 11], "batter_id": [20, 21, 22],
        "pitcher_hand": [1, 1, 2], "batter_hand": [1, 2, 1],
        "pitcher_team_id": [1, 1, 2], "batter_team_id": [2, 2, 1],
        "asof_pitcher_n": [10, 20, 5], "asof_pitcher_success_rate": [0.5, 0.55, 0.4],
        "asof_pitcher_reverse_rate": [0.1] * 3, "asof_pitcher_middle_rate": [0.2] * 3,
        "asof_pitcher_ball_rate": [0.3] * 3, "asof_pitcher_strike_rate": [0.4] * 3,
        "asof_pitcher_prev1_game_success_rate": [0.5] * 3,
        "asof_pitcher_prev3_game_success_rate": [0.5] * 3,
        "asof_pitcher_prev5_game_success_rate": [0.5] * 3,
        "asof_pitcher_prev1_game_middle_rate": [0.2] * 3,
        "asof_pitcher_prev3_game_middle_rate": [0.2] * 3,
        "asof_pitcher_prev5_game_middle_rate": [0.2] * 3,
        "asof_batter_n": [5, 6, 7], "asof_batter_success_rate": [0.5] * 3,
        "asof_batter_middle_rate": [0.2] * 3, "asof_pitcher_pitchmix_n": [10] * 3,
        "asof_pitcher_fastball_rate": [0.5] * 3,
        "asof_pitcher_breaking_rate": [0.3] * 3,
        "asof_pitcher_offspeed_rate": [0.2] * 3,
        "control_success": [1, 0, 1],
    }
    return pd.DataFrame(columns)


def test_features_exclude_identity_month_and_are_row_order_invariant():
    rows = _rows()
    strength = {"pitcher": 300.0, "pitcher_batter_hand": 200.0, "pitcher_count": 150.0}
    bank = _bank(rows.iloc[:2], strength)
    one = _feature_frame(rows.iloc[[2]].reset_index(drop=True), bank)
    two = _feature_frame(rows.iloc[[2, 1]].reset_index(drop=True), bank).iloc[[0]].reset_index(drop=True)
    assert "pitcher_id" not in one and "batter_id" not in one and "game_month" not in one
    pd.testing.assert_frame_equal(one, two)


def test_route_protection_is_exact():
    parent = np.array([0.4, 0.5, 0.6])
    direct = np.array([0.7, 0.2, 0.3])
    domain = np.array(["R_CORE", "R_ANCHOR", "F"])
    candidate = _apply(parent, direct, domain, ("R_CORE",), 0.1)
    assert candidate[0] != parent[0]
    assert np.array_equal(candidate[1:], parent[1:])
