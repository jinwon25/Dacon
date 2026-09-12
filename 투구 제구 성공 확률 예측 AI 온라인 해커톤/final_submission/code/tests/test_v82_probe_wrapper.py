import numpy as np
import pandas as pd

from src.archive.v82_probe_wrapper import blend_predictions, probe_mask


def test_probe_mask_is_regular_non_anchor_only() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "R", "F"],
            "pitcher_team_id": [1, 13, 1, 1],
            "batter_team_id": [2, 2, 13, 2],
        }
    )
    assert probe_mask(frame).tolist() == [True, False, False, False]


def test_blend_changes_only_active_rows() -> None:
    champion = np.array([0.4, 0.5, 0.6])
    strict = np.array([0.6, 0.3, 0.8])
    output = blend_predictions(champion, strict, np.array([True, False, True]))
    assert np.allclose(output, [0.42, 0.5, 0.62])
