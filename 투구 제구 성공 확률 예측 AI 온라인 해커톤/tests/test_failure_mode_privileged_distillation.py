import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import reconstruct_failure_mode


def test_reconstruct_failure_mode_from_next_asof_row():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 1, 1, 1],
            "asof_pitcher_n": [0, 1, 2, 3, 4],
            "asof_pitcher_reverse_rate": [np.nan, 1.0, 0.5, 1 / 3, 0.25],
            "asof_pitcher_middle_rate": [np.nan, 0.0, 0.0, 1 / 3, 0.25],
            "asof_pitcher_ball_rate": [np.nan, 0.0, 0.5, 1 / 3, 0.25],
            "asof_pitcher_strike_rate": [np.nan, 0.0, 0.0, 0.0, 0.25],
        }
    )
    # Rows reconstruct respectively reverse/bad, ball, middle/bad, strike;
    # the final pitcher row has no next cumulative state.
    assert reconstruct_failure_mode(frame).tolist() == [0, 1, 0, 2, -1]
