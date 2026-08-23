import numpy as np
import pandas as pd

from src.archive.v97_conditional_direct_forward import _bank
from src.archive.v109_v104_feature_component import build_bank
from src.archive.v109_v104_probe_wrapper import apply_v104, stability_mask


def test_standalone_bank_matches_research_bank() -> None:
    history = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2],
            "batter_hand": [1, 2, 1, 2],
            "balls_before": [0, 1, 2, 3],
            "strikes_before": [0, 1, 2, 2],
            "control_success": [1, 0, 1, 1],
        }
    )
    strengths = {
        "pitcher": 300.0,
        "pitcher_batter_hand": 200.0,
        "pitcher_count": 150.0,
    }
    expected = _bank(history, strengths)
    actual = build_bank(history, strengths)
    assert actual["global"] == expected["global"]
    for name in ("pitcher", "hand", "count"):
        pd.testing.assert_frame_equal(actual[name], expected[name])


def test_majority_mask_and_route_protect_noneligible_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 1, 1],
            "batter_team_id": [2, 2, 2],
            "balls_before": [0, 0, 0],
            "strikes_before": [0, 1, 0],
            "asof_pitcher_n": [1200, 10, 1200],
            "pitcher_hand": [1, 1, 1],
            "batter_hand": [1, 2, 1],
            "inning": [1, 1, 1],
        }
    )
    assert stability_mask(frame).tolist() == [True, False, True]
    parent = np.array([0.4, 0.4, 0.4])
    candidate = apply_v104(
        parent,
        frame,
        np.array([0.1, 0.1, 0.1]),
        np.array([0.02, 0.02, 0.02]),
    )
    assert candidate[0] != parent[0]
    assert candidate[1] == parent[1]
    assert candidate[2] == parent[2]
