from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.compose_multiyear_g2_q1_dose_candidate import (
    active_month_mask,
    apply_month_route,
    q1_mask,
)


def test_apply_month_route_changes_only_february() -> None:
    index = pd.DatetimeIndex(
        ["2025-01-01", "2025-02-28", "2025-03-31", "2025-12-01"]
    )
    anchor = np.asarray([1.0, 2.0, 3.0, 4.0])
    treatment = np.asarray([10.0, 20.0, 30.0, 40.0])
    assert q1_mask(index).tolist() == [True, True, True, False]
    assert active_month_mask(index).tolist() == [False, True, False, False]
    assert apply_month_route(anchor, treatment, index).tolist() == [
        1.0,
        20.0,
        3.0,
        4.0,
    ]


def test_apply_q1_route_rejects_misalignment() -> None:
    with pytest.raises(ValueError, match="align"):
        apply_month_route(
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
            pd.date_range("2025-01-01", periods=2),
        )
