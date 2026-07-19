from __future__ import annotations

import pandas as pd
import pytest

from experiments.build_noaa_gefs_spread_features import interpolate_to_targets


def test_interpolation_uses_lead_brackets_per_grid() -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2025-01-01 02:00:00"],
            "data_available_kst_dtm": ["2024-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.5],
            "longitude": [128.75],
        }
    )
    plan = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2025-01-01 02:00:00"],
            "data_available_kst_dtm": ["2024-12-31 13:00:00"],
            "initialization_utc": ["2024-12-30 18:00:00+00:00"],
            "lower_lead_hour": [21],
            "upper_lead_hour": [24],
            "upper_weight": [2 / 3],
        }
    )
    decoded = pd.DataFrame(
        {
            "initialization_utc": [
                "2024-12-30 18:00:00+00:00",
                "2024-12-30 18:00:00+00:00",
            ],
            "lead_hour": [21, 24],
            "grid_id": [1, 1],
            "gefs_u10_spread": [1.0, 4.0],
            "gefs_v10_spread": [2.0, 5.0],
        }
    )
    output = interpolate_to_targets(metadata, plan, decoded)
    assert output.loc[0, "gefs_u10_spread"] == pytest.approx(3.0)
    assert output.loc[0, "gefs_v10_spread"] == pytest.approx(4.0)
    assert output.loc[0, "gefs_uv10_spread_norm"] == pytest.approx(5.0)


def test_mean_interpolation_uses_mean_columns() -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2025-01-01 01:00:00"],
            "data_available_kst_dtm": ["2024-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.5],
            "longitude": [128.75],
        }
    )
    plan = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2025-01-01 01:00:00"],
            "data_available_kst_dtm": ["2024-12-31 13:00:00"],
            "initialization_utc": ["2024-12-30 18:00:00+00:00"],
            "lower_lead_hour": [21],
            "upper_lead_hour": [24],
            "upper_weight": [1 / 3],
        }
    )
    decoded = pd.DataFrame(
        {
            "initialization_utc": [
                "2024-12-30 18:00:00+00:00",
                "2024-12-30 18:00:00+00:00",
            ],
            "lead_hour": [21, 24],
            "grid_id": [1, 1],
            "gefs_u10_mean": [2.0, 5.0],
            "gefs_v10_mean": [1.0, 4.0],
        }
    )
    output = interpolate_to_targets(metadata, plan, decoded, statistic="mean")
    assert output.loc[0, "gefs_u10_mean"] == pytest.approx(3.0)
    assert output.loc[0, "gefs_v10_mean"] == pytest.approx(2.0)
