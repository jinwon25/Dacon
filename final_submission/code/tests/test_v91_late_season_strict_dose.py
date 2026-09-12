from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v91_late_season_strict_dose import apply_late_strict_weight


def test_late_strict_replaces_only_august_r_core() -> None:
    frame = pd.DataFrame(
        {
            "domain3": ["R_CORE", "R_CORE", "R_ANCHOR", "F"],
            "game_month": [7, 8, 9, 10],
        }
    )
    parent = np.asarray([0.4, 0.4, 0.4, 0.4])
    strict = np.asarray([0.6, 0.6, 0.6, 0.6])
    current = parent.copy()
    current[:2] = parent[:2] + 0.1 * (strict[:2] - parent[:2])
    candidate, active = apply_late_strict_weight(
        frame, current, parent, strict, 0.15
    )
    assert active.tolist() == [False, True, False, False]
    assert candidate[0] == current[0]
    assert np.isclose(candidate[1], 0.43)
    assert np.allclose(candidate[2:], current[2:])
