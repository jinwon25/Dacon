import numpy as np
import pandas as pd

from src.archive.v304_current_appearance_renewal_audit import (
    MAX_LENGTH,
    attach_appearance_state,
    renewal_posterior_table,
)


def test_attach_appearance_state_counts_pitcher_age_within_game() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"TRAIN_{index:07d}" for index in range(1, 7)],
            "season": [2021] * 6,
            "game_type": ["R"] * 6,
            "inning": [1, 1, 1, 1, 1, 1],
            "top_bottom": ["T", "T", "T", "T", "B", "B"],
            "pitcher_id": [10, 10, 11, 10, 12, 12],
            "asof_pitcher_n": [0, 1, 0, 2, 0, 1],
        }
    )
    rows, appearances = attach_appearance_state(frame)
    assert rows["_appearance_age"].tolist() == [0, 1, 0, 2, 0, 1]
    assert sorted(appearances["pitches"].tolist()) == [1, 2, 3]


def test_renewal_posterior_is_exact_for_constant_length() -> None:
    pmf = np.zeros(MAX_LENGTH + 1, dtype=np.float64)
    pmf[5] = 1.0
    mean, sd, p20, p60 = renewal_posterior_table(pmf, 19)
    assert np.allclose(mean, np.arange(20) % 5)
    assert np.allclose(sd, 0.0)
    assert np.allclose(p20, 0.0)
    assert np.allclose(p60, 0.0)
