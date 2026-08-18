import pandas as pd

from src.context_residual_screen import add_fixed_context, context_groups


def test_fixed_context_does_not_require_target():
    frame = pd.DataFrame(
        {
            "balls_before": [3], "strikes_before": [2], "inning": [8],
            "score_diff_pitcher_team": [-7], "li": [2.5],
            "home_win_expectancy": [65.0], "asof_pitcher_n": [255],
            "asof_batter_n": [0], "asof_pitcher_pitchmix_n": [31],
            "asof_pitcher_success_rate": [0.51], "asof_pitcher_reverse_rate": [0.2],
            "asof_pitcher_middle_rate": [0.1], "asof_pitcher_ball_rate": [0.4],
            "asof_pitcher_strike_rate": [0.3], "asof_batter_success_rate": [None],
            "asof_batter_middle_rate": [None], "asof_pitcher_fastball_rate": [0.6],
            "asof_pitcher_breaking_rate": [0.3], "asof_pitcher_offspeed_rate": [0.1],
            "asof_pitcher_prev1_game_success_rate": [0.6],
            "asof_pitcher_prev3_game_success_rate": [0.5],
            "asof_pitcher_prev5_game_success_rate": [0.4],
        }
    )
    enriched = add_fixed_context(frame)
    assert enriched.loc[0, "count_state"] == "3-2"
    assert enriched.loc[0, "inning_group"] == "late"
    assert enriched.loc[0, "score_group"] == "-4"
    assert "control_success" not in enriched
    assert all("control_success" not in columns for columns in context_groups().values())
