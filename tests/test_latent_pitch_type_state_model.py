import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import reconstruct_current_pitch_type


def test_reconstruct_current_pitch_type_from_next_snapshot() -> None:
    rows = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "pitcher_id": [10, 20, 10, 10],
            "asof_pitcher_pitchmix_n": [0, 0, 1, 2],
            "asof_pitcher_fastball_rate": [np.nan, np.nan, 1.0, 0.5],
            "asof_pitcher_breaking_rate": [np.nan, np.nan, 0.0, 0.5],
            "asof_pitcher_offspeed_rate": [np.nan, np.nan, 0.0, 0.0],
        }
    )

    actual = reconstruct_current_pitch_type(rows)

    assert actual.tolist() == [0, -1, 1, -1]
