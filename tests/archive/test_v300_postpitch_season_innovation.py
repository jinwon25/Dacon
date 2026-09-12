import numpy as np
import pandas as pd

from src.archive.v300_postpitch_season_innovation import season_innovation, terminal_snapshot


def test_terminal_snapshot_includes_last_observed_pitch():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2],
            "asof_pitcher_n": [10, 11, 4],
            "asof_pitcher_success_rate": [0.5, 6 / 11, 0.25],
            "control_success": [1, 0, 1],
        }
    )
    snapshot = terminal_snapshot(
        frame, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    assert snapshot[1] == (12.0, 6.0)
    assert snapshot[2] == (5.0, 2.0)


def test_season_innovation_is_zero_at_exact_opening():
    frame = pd.DataFrame(
        {
            "pitcher_id": [1],
            "asof_pitcher_n": [12.0],
            "asof_pitcher_success_rate": [0.5],
        }
    )
    delta, season_n, opening_n = season_innovation(
        frame,
        {1: (12.0, 6.0)},
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        100.0,
    )
    np.testing.assert_allclose(delta, 0.0)
    np.testing.assert_allclose(season_n, 0.0)
    np.testing.assert_allclose(opening_n, 12.0)
