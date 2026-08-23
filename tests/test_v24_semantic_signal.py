import numpy as np
import pandas as pd

from src.champion.v24_semantic_signal_screen import level_history_signal, shape_signal


def test_shape_signal_is_centered_within_level_and_row_local():
    source = pd.DataFrame(
        {
            "game_type": ["R"] * 4 + ["F"] * 4,
            "li": [0.1, 0.2, 2.0, 3.0] * 2,
            "score_diff_pitcher_team": [0] * 8,
            "control_success": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )
    audit = source.iloc[[0, 3, 4, 7]].drop(columns="control_success")
    together = shape_signal(source, audit, "li5", 1.0)
    separately = np.asarray(
        [shape_signal(source, audit.iloc[[i]], "li5", 1.0)[0] for i in range(len(audit))]
    )
    np.testing.assert_allclose(together, separately)


def test_level_history_signal_uses_matching_level_only():
    history = pd.DataFrame(
        {
            "season": [2022] * 8,
            "game_type": ["R"] * 4 + ["F"] * 4,
            "pitcher_id": [1, 1, 2, 2] * 2,
            "batter_id": [10, 11, 10, 11] * 2,
            "control_success": [1, 1, 0, 0, 0, 0, 1, 1],
        }
    )
    audit = pd.DataFrame(
        {
            "game_type": ["R", "F"],
            "pitcher_id": [1, 1],
            "batter_id": [10, 10],
        }
    )
    signal = level_history_signal(history, audit, half_life=1.0, alpha=1.0)
    assert signal[0] > signal[1]
