import pandas as pd

from src.champion.v25_pitch_type_id_student import feature_frame


def test_feature_frame_contains_pitcher_count_interactions() -> None:
    row = {column: 0 for column in (
        "game_month", "inning", "balls_before", "strikes_before", "outs_before",
        "score_diff_pitcher_team", "num_runners_on", "li", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    )}
    row.update({
        "pitcher_id": 7, "batter_id": 9, "pitcher_team_id": 1,
        "batter_team_id": 2, "pitcher_hand": "R", "batter_hand": "L",
        "game_type": "R", "domain3": "R_CORE", "base_state": "___",
        "top_bottom": "T",
    })
    features, categorical = feature_frame(pd.DataFrame([row]))
    assert "cat__pitcher_count" in categorical
    assert "cat__pitcher_count_hand" in categorical
    assert str(features["cat__pitcher_count"].iloc[0]) == "7|0|0"
