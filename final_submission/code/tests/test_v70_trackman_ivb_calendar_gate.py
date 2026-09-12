from __future__ import annotations

import pandas as pd

from src.archive.v70_trackman_ivb_calendar_gate import calendar_mask


def test_calendar_mask_uses_only_april_through_september() -> None:
    frame = pd.DataFrame({"game_month": [3, 4, 7, 9, 10]})
    assert calendar_mask(frame).tolist() == [False, True, True, True, False]
