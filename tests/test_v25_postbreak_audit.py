from __future__ import annotations

import numpy as np
import pandas as pd

from src.v25_postbreak_anchor_audit import _evaluate


def test_evaluate_changes_only_anchor_rows() -> None:
    frame = pd.DataFrame(
        {
            "target": [1.0, 0.0, 0.0, 1.0],
            "v22": [0.4, 0.6, 0.6, 0.4],
            "domain3": ["R_ANCHOR", "R_ANCHOR", "R_CORE", "F"],
            "game_month": [8, 9, 8, 9],
            "pitcher_hand": [1, 2, 1, 2],
            "batter_hand": [2, 1, 2, 1],
            "count_state": ["0-0", "1-1", "0-0", "1-1"],
        }
    )
    direct = np.asarray([0.9, 0.1, 0.1, 0.9])
    result = _evaluate(frame, direct)
    assert result["gain"] > 0.0
    assert result["applied_rows"] == 2
    assert result["positive_active_month_fraction"] == 1.0
