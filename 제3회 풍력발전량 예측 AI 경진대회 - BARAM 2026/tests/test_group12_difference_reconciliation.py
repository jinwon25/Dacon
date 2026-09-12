from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.group12_difference_reconciliation import (
    build_pairwise_features,
    reconcile_pair,
)


def test_reconcile_pair_preserves_sum_and_bounds() -> None:
    first = np.asarray([0.0, 10_000.0, 21_600.0])
    second = np.asarray([21_600.0, 11_000.0, 0.0])
    expert = np.asarray([1.0, -0.2, -1.0])
    adjusted_first, adjusted_second, movement = reconcile_pair(
        first,
        second,
        expert,
        weight=0.10,
        movement_cap_ratio=0.02,
    )
    assert np.allclose(
        adjusted_first + adjusted_second,
        first + second,
    )
    assert np.all((adjusted_first >= 0.0) & (adjusted_first <= 21_600.0))
    assert np.all((adjusted_second >= 0.0) & (adjusted_second <= 21_600.0))
    assert np.max(np.abs(movement)) <= 0.02 * 21_600.0


def test_pairwise_features_are_symmetric_mean_and_difference() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    features = pd.DataFrame(
        {
            "ldaps__kpx_group_1__hub_ws117__idw": [4.0, 8.0],
            "ldaps__kpx_group_2__hub_ws117__idw": [2.0, 10.0],
            "gfs__kpx_group_1__hub_ws117__idw": [5.0, 9.0],
            "gfs__kpx_group_2__hub_ws117__idw": [3.0, 7.0],
            "hour": [0.0, 1.0],
        },
        index=index,
    )
    # The production guard expects at least 30 fields, so add paired dummy
    # weather channels while retaining an explicit assertion on the real pair.
    for number in range(20):
        features[f"x{number}__kpx_group_1__v__idw"] = float(number)
        features[f"x{number}__kpx_group_2__v__idw"] = float(number + 1)
    result = build_pairwise_features(features)
    stem = "ldaps__pair__hub_ws117__idw"
    assert np.allclose(result[f"mean__{stem}"], [3.0, 9.0])
    assert np.allclose(result[f"delta__{stem}"], [2.0, -2.0])
