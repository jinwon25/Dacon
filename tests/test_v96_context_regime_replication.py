from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v96_context_regime_replication import early_late_masks


def test_early_late_masks_are_fixed_and_row_local() -> None:
    frame = pd.DataFrame({"game_month": [3, 7, 8, 10]})
    early, late = early_late_masks(frame)
    np.testing.assert_array_equal(early, [True, True, False, False])
    np.testing.assert_array_equal(late, [False, False, True, True])
    order = np.array([2, 0, 3, 1])
    shuffled_early, shuffled_late = early_late_masks(frame.iloc[order])
    np.testing.assert_array_equal(shuffled_early, early[order])
    np.testing.assert_array_equal(shuffled_late, late[order])
