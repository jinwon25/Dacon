from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.build_noaa_cfsv2_features import interpolate_to_targets


def test_interpolate_cfsv2_six_hour_leads() -> None:
    issue = pd.Timestamp("2023-12-31 13:00:00")
    initialization = pd.Timestamp("2023-12-29 18:00:00", tz="UTC")
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": [issue],
            "grid_id": [1],
            "latitude": [37.25],
            "longitude": [129.0],
        }
    )
    join = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": [issue],
            "initialization_utc": [initialization],
            "lower_lead_hour": [42],
            "upper_lead_hour": [48],
            "upper_weight": [2.0 / 3.0],
        }
    )
    decoded = pd.DataFrame(
        {
            "initialization_utc": [initialization, initialization],
            "lead_hour": [42, 48],
            "grid_id": [1, 1],
            "cfsv2_u10": [1.0, 4.0],
            "cfsv2_v10": [2.0, 8.0],
        }
    )
    result = interpolate_to_targets(metadata, join, decoded)
    assert result.loc[0, "cfsv2_u10"] == pytest.approx(3.0)
    assert result.loc[0, "cfsv2_v10"] == pytest.approx(6.0)
    assert result.loc[0, "cfsv2_speed10"] == pytest.approx(np.hypot(3.0, 6.0))


def test_interpolate_cfsv2_fails_on_missing_bracket() -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.25],
            "longitude": [129.0],
        }
    )
    join = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "initialization_utc": ["2023-12-29 18:00:00+00:00"],
            "lower_lead_hour": [42],
            "upper_lead_hour": [48],
            "upper_weight": [0.5],
        }
    )
    decoded = pd.DataFrame(
        {
            "initialization_utc": ["2023-12-29 18:00:00+00:00"],
            "lead_hour": [42],
            "grid_id": [1],
            "cfsv2_u10": [1.0],
            "cfsv2_v10": [2.0],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        interpolate_to_targets(metadata, join, decoded)


def test_interpolate_deduplicates_identical_shared_source_messages() -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.25],
            "longitude": [129.0],
        }
    )
    join = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "initialization_utc": ["2023-12-29 18:00:00+00:00"],
            "lower_lead_hour": [42],
            "upper_lead_hour": [48],
            "upper_weight": [0.5],
        }
    )
    decoded = pd.DataFrame(
        {
            "initialization_utc": [
                "2023-12-29 18:00:00+00:00",
                "2023-12-29 18:00:00+00:00",
                "2023-12-29 18:00:00+00:00",
            ],
            "lead_hour": [42, 42, 48],
            "grid_id": [1, 1, 1],
            "cfsv2_u10": [1.0, 1.0, 3.0],
            "cfsv2_v10": [2.0, 2.0, 6.0],
        }
    )
    result = interpolate_to_targets(metadata, join, decoded)
    assert result.loc[0, "cfsv2_u10"] == pytest.approx(2.0)
    assert result.loc[0, "cfsv2_v10"] == pytest.approx(4.0)
