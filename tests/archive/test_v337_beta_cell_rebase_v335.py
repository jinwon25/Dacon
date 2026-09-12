import pandas as pd

from src.archive import v337_beta_cell_rebase_v335 as v337


def test_fixed_cell_mask_uses_only_the_frozen_group() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [12, 13, 12],
            "batter_team_id": [14, 14, 14],
            "asof_pitcher_n": [200.0, 200.0, 200.0],
            "asof_pitcher_pitchmix_n": [200.0, 200.0, 200.0],
            "asof_pitcher_fastball_rate": [0.4, 0.4, 0.4],
            "asof_pitcher_breaking_rate": [0.3, 0.3, 0.3],
            "asof_pitcher_offspeed_rate": [0.3, 0.3, 0.3],
            "asof_pitcher_reverse_rate": [0.1, 0.1, 0.1],
            "asof_pitcher_middle_rate": [0.2, 0.2, 0.2],
            "asof_pitcher_ball_rate": [0.4, 0.4, 0.4],
            "asof_pitcher_strike_rate": [0.3, 0.3, 0.3],
            "asof_pitcher_prev1_game_success_rate": [0.5, 0.5, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.5, 0.5, 0.5],
            "balls_before": [1, 1, 1],
            "strikes_before": [1, 1, 1],
            "pitcher_hand": [0, 0, 0],
            "batter_hand": [1, 1, 1],
            "li": [1.0, 1.0, 1.0],
        }
    )
    assert v337.fixed_cell_mask(frame).tolist() == [True, False, False]
