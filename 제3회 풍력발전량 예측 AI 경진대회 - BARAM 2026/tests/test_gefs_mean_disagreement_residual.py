from __future__ import annotations

import pandas as pd

from experiments.gefs_mean_disagreement_residual import build_disagreement_features


def test_disagreement_builder_rejects_unaligned_rows(tmp_path) -> None:
    mean = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.5],
            "longitude": [128.75],
            "gefs_u10_mean": [1.0],
            "gefs_v10_mean": [2.0],
        }
    )
    gfs = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 02:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
            "grid_id": [1],
            "latitude": [37.5],
            "longitude": [128.75],
            "heightAboveGround_10_10u": [1.0],
            "heightAboveGround_10_10v": [2.0],
            "heightAboveGround_80_u": [1.0],
            "heightAboveGround_80_v": [2.0],
            "heightAboveGround_100_100u": [1.0],
            "heightAboveGround_100_100v": [2.0],
        }
    )
    mean_path = tmp_path / "mean.csv"
    gfs_path = tmp_path / "gfs.csv"
    mean.to_csv(mean_path, index=False, encoding="utf-8-sig")
    gfs.to_csv(gfs_path, index=False, encoding="utf-8-sig")
    try:
        build_disagreement_features(mean_path, gfs_path)
    except ValueError as exc:
        assert "not exactly aligned" in str(exc)
    else:
        raise AssertionError("unaligned mean/GFS rows must fail closed")
