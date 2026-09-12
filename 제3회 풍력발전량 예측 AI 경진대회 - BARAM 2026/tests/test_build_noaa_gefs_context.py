from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.build_noaa_gefs_context import (
    aggregate_gefs_spread,
    align_to_primary_context,
)


def _source(times: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_position, forecast in enumerate(times):
        for grid_id in range(1, 10):
            rows.append(
                {
                    "forecast_kst_dtm": forecast,
                    "data_available_kst_dtm": forecast
                    - pd.Timedelta(hours=12),
                    "grid_id": grid_id,
                    "gefs_u10_spread": time_position + grid_id / 10.0,
                    "gefs_v10_spread": time_position + grid_id / 20.0,
                    "gefs_uv10_spread_norm": time_position + grid_id / 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_gefs_spread_has_fixed_spatial_contract() -> None:
    times = [pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-01 02:00")]
    features, issue = aggregate_gefs_spread(_source(times))

    assert features.shape == (2, 21)
    assert features.columns.str.startswith("kma_um_ctx_gefs_spread_").all()
    assert np.isclose(
        features.loc[times[0], "kma_um_ctx_gefs_spread_u10__centre"],
        0.5,
    )
    assert np.isclose(
        features.loc[times[0], "kma_um_ctx_gefs_spread_u10__east_west"],
        0.2,
    )
    assert issue.loc[times[1]] == times[1] - pd.Timedelta(hours=12)


def test_align_permits_one_causal_leading_boundary_gap() -> None:
    times = [pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-01 02:00")]
    features, issue = aggregate_gefs_spread(_source(times))
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": [
                pd.Timestamp("2024-01-01 00:00"),
                *times,
            ],
            "data_available_kst_dtm": [
                pd.Timestamp("2023-12-31 13:00"),
                pd.Timestamp("2023-12-31 13:00"),
                pd.Timestamp("2023-12-31 14:00"),
            ],
        }
    )

    aligned, aligned_issue, imputed = align_to_primary_context(
        features, issue, primary
    )

    assert aligned.shape == (3, 21)
    assert imputed == ["2024-01-01T00:00:00"]
    assert np.allclose(aligned.iloc[0].to_numpy(), 0.0)
    assert aligned_issue.iloc[0] <= primary.loc[0, "data_available_kst_dtm"]


def test_aggregate_rejects_incomplete_stencil() -> None:
    frame = _source([pd.Timestamp("2024-01-01 01:00")]).iloc[:-1]
    with pytest.raises(ValueError, match="all nine grids"):
        aggregate_gefs_spread(frame)
