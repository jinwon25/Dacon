from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.kma_spatial_context_screen import (
    build_spatial_features,
    interpolate_expert,
)


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kma_um_ctx_u10_r0": values,
            "doy_sin": [0.0] * len(values),
        },
        index=pd.date_range("2024-01-01", periods=len(values), freq="h"),
    )


def test_build_spatial_features_adds_mean_spread_and_gradient() -> None:
    centre = _frame([2.0, 4.0])
    east = _frame([4.0, 8.0])
    west = _frame([0.0, 2.0])
    result = build_spatial_features(centre, east, west)
    column = "kma_um_ctx_u10_r0"
    np.testing.assert_allclose(
        result[f"kma_spatial_mean__{column}"],
        np.asarray([2.0, 14.0 / 3.0]),
        rtol=1e-6,
    )
    assert result[f"kma_spatial_east_minus_west__{column}"].tolist() == [
        4.0,
        6.0,
    ]
    expected_std = np.std(np.asarray([[0.0, 2.0, 4.0], [2.0, 4.0, 8.0]]), axis=1)
    np.testing.assert_allclose(
        result[f"kma_spatial_std__{column}"], expected_std, rtol=1e-6
    )


def test_build_spatial_features_rejects_misaligned_index() -> None:
    centre = _frame([1.0, 2.0])
    east = _frame([1.0, 2.0])
    east.index = east.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="indexes differ"):
        build_spatial_features(centre, east, _frame([1.0, 2.0]))


def test_interpolate_expert_is_bounded_replacement() -> None:
    centre = np.asarray([100.0, 200.0])
    spatial = np.asarray([300.0, 100.0])
    np.testing.assert_allclose(
        interpolate_expert(centre, spatial, 0.5),
        np.asarray([200.0, 150.0]),
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_expert(centre, spatial, 1.5)
