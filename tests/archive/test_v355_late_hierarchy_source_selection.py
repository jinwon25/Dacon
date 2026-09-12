import numpy as np
import pandas as pd

from src.archive.v355_late_hierarchy_source_selection import evaluate


def test_evaluate_uses_fixed_late_anchor_blend():
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "R"],
            "pitcher_team_id": [13, 13, 13],
            "batter_team_id": [16, 16, 16],
            "game_month": [7, 8, 8],
            "control_success": [0, 1, 0],
            "pitcher_id": [1, 1, 1],
            "batter_id": [2, 2, 2],
        }
    )
    parent = np.asarray([0.4, 0.4, 0.4])
    raw = np.asarray([0.8, 0.8, 0.8])
    candidate, active, _metrics = evaluate(frame, parent, raw)
    np.testing.assert_array_equal(active, [False, True, True])
    np.testing.assert_allclose(candidate, [0.4, 0.48, 0.48])
