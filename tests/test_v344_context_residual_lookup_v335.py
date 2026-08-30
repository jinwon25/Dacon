import numpy as np
import pandas as pd

from src.archive import v344_context_residual_lookup_v335 as v344


def test_lookup_is_row_local_and_identity_free() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 13, 1],
            "batter_team_id": [2, 2, 2],
            "balls_before": [1, 1, 1],
            "strikes_before": [2, 2, 2],
            "base_state": ["---", "---", "---"],
            "pitcher_hand": [1, 1, 1],
            "batter_hand": [1, 1, 1],
            "num_runners_on": [0, 0, 0],
            "li": [1.0, 1.0, 1.0],
        }
    )
    columns = v344.SCHEMAS["count_hand"]
    table = pd.DataFrame(
        {"balls_before": [1], "strikes_before": [2], "same_hand": [1], "size": [100], "correction": [0.04]}
    )
    candidate, active, correction = v344.apply_lookup(
        frame, np.array([0.5, 0.5, 0.5]), table, columns, dose=0.25
    )
    assert np.allclose(correction, [0.04, 0.04, 0.04])
    assert active.tolist() == [True, False, False]
    assert np.allclose(candidate, [0.51, 0.5, 0.5])
