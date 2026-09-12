import numpy as np
import pandas as pd

from src.archive.v378_destination_transition_effect_v345 import _apply, _count_key


def test_count_key_is_row_local():
    frame = pd.DataFrame({"balls_before": [0, 3], "strikes_before": [0, 2]})
    assert _count_key(frame).tolist() == ["0-0", "3-2"]


def test_apply_protects_anchor_futures_and_same_pitchers():
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F", "R"],
            "pitcher_team_id": [12, 13, 12, 12],
            "batter_team_id": [14, 14, 14, 14],
        }
    )
    parent = np.full(4, 0.5)
    signal = np.full(4, 0.04)
    status = np.asarray(["SWITCH", "SWITCH", "SWITCH", "SAME"])
    candidate, active = _apply(
        frame, parent, signal, status, frozenset(("SWITCH",))
    )
    assert active.tolist() == [True, False, False, False]
    assert np.allclose(candidate, [0.51, 0.5, 0.5, 0.5])
