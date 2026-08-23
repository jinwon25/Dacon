import pandas as pd

from src.champion.v23_postbreak_gam_screen import _columns


def test_columns_separate_rate_splines_from_linear_context():
    frame = pd.DataFrame(
        {
            "control_success": [1],
            "pitcher_id": [3],
            "game_type": ["R"],
            "asof_pitcher_success_rate": [0.5],
            "asof_pitcher_n": [100],
            "li": [1.0],
        }
    )
    rates, linear = _columns(frame)
    assert rates == ["asof_pitcher_success_rate"]
    assert linear == ["asof_pitcher_n", "li"]
