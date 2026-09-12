import numpy as np
import pandas as pd

from src.archive.v189_pressure_profile_route_screen import route_library


def test_route_library_is_row_local() -> None:
    frame = pd.DataFrame(
        {
            "li": [2.0, 0.5], "inning": [8, 2],
            "score_diff_pitcher_team": [0, 4], "balls_before": [3, 0],
            "strikes_before": [1, 2], "num_runners_on": [1, 0],
            "runner_on_2b": [1, 0], "runner_on_3b": [0, 0],
        }
    )
    routes = route_library()
    np.testing.assert_array_equal(routes["high_li"](frame), [True, False])
    np.testing.assert_array_equal(routes["traffic"](frame), [True, False])
    np.testing.assert_array_equal(routes["two_strike"](frame), [False, True])
