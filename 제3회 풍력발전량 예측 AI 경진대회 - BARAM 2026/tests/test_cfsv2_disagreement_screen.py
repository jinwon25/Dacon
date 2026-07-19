from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.cfsv2_disagreement_screen import build_disagreement_features


def test_build_cfsv2_gfs_disagreement_features(tmp_path: Path) -> None:
    time = pd.Timestamp("2024-01-01 01:00:00")
    issue = pd.Timestamp("2023-12-31 13:00:00")
    cfs_rows = []
    gfs_rows = []
    for grid_id in range(1, 10):
        latitude = 37.5 - 0.25 * ((grid_id - 1) // 3)
        longitude = 128.75 + 0.25 * ((grid_id - 1) % 3)
        cfs_rows.append(
            {
                "forecast_kst_dtm": time,
                "data_available_kst_dtm": issue,
                "grid_id": grid_id,
                "latitude": latitude,
                "longitude": longitude,
                "cfsv2_u10": 3.0,
                "cfsv2_v10": 4.0,
                "cfsv2_speed10": 5.0,
            }
        )
        gfs_rows.append(
            {
                "forecast_kst_dtm": time,
                "data_available_kst_dtm": issue,
                "grid_id": grid_id,
                "latitude": latitude,
                "longitude": longitude,
                "heightAboveGround_10_10u": 1.0,
                "heightAboveGround_10_10v": 0.0,
                "heightAboveGround_100_100u": 2.0,
                "heightAboveGround_100_100v": 0.0,
            }
        )
    cfs_path = tmp_path / "cfs.csv"
    gfs_path = tmp_path / "gfs.csv"
    pd.DataFrame(cfs_rows).to_csv(cfs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(gfs_rows).to_csv(gfs_path, index=False, encoding="utf-8-sig")
    result = build_disagreement_features(cfs_path, gfs_path)
    assert len(result) == 1
    assert result.loc[time, "delta_u10_mean"] == pytest.approx(2.0)
    assert result.loc[time, "delta_v10_mean"] == pytest.approx(4.0)
    assert result.loc[time, "vector_disagreement_center"] == pytest.approx(
        np.hypot(2.0, 4.0)
    )
    assert result.loc[time, "speed_disagreement_center"] == pytest.approx(4.0)
    assert result.loc[time, "cfsv2_vs_gfs100_speed_ratio_center"] == pytest.approx(
        2.5
    )


def test_disagreement_builder_fails_on_missing_gfs_hour(tmp_path: Path) -> None:
    cfs = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.5],
            "longitude": [128.75],
            "cfsv2_u10": [1.0],
            "cfsv2_v10": [2.0],
            "cfsv2_speed10": [np.hypot(1.0, 2.0)],
        }
    )
    cfs_path = tmp_path / "cfs.csv"
    gfs_path = tmp_path / "gfs.csv"
    cfs.to_csv(cfs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=[
        "forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude",
        "longitude", "heightAboveGround_10_10u", "heightAboveGround_10_10v",
        "heightAboveGround_100_100u", "heightAboveGround_100_100v",
    ]).to_csv(gfs_path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="not exactly aligned"):
        build_disagreement_features(cfs_path, gfs_path)
