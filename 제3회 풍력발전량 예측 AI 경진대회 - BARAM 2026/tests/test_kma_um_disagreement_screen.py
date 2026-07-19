from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.kma_um_disagreement_screen import (
    build_kma_gfs_disagreement_features,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    issue = pd.Timestamp("2023-12-31 13:00:00")
    target = pd.Timestamp("2024-01-01 01:00:00")
    kma = pd.DataFrame(
        {
            "forecast_kst_dtm": [target, target],
            "data_available_kst_dtm": [issue, issue],
            "point_id": [1, 2],
            "initialization_utc": ["2023-12-30T12:00:00+00:00"] * 2,
            "public_availability_utc": ["2023-12-31T00:00:00+00:00"] * 2,
            "kma_um_u10": [3.0, 5.0],
            "kma_um_v10": [4.0, 12.0],
            "kma_um_speed10": [5.0, 13.0],
        }
    )
    gfs_rows = []
    for grid_id in range(1, 10):
        gfs_rows.append(
            {
                "forecast_kst_dtm": target,
                "data_available_kst_dtm": issue,
                "grid_id": grid_id,
                "heightAboveGround_10_10u": 2.0 if grid_id == 5 else 1.0,
                "heightAboveGround_10_10v": 0.0,
                "heightAboveGround_100_100u": 4.0,
                "heightAboveGround_100_100v": 0.0,
            }
        )
    kma_path = tmp_path / "kma.csv"
    gfs_path = tmp_path / "gfs.csv"
    kma.to_csv(kma_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(gfs_rows).to_csv(gfs_path, index=False, encoding="utf-8-sig")
    return kma_path, gfs_path


def test_build_kma_gfs_disagreement_features(tmp_path) -> None:
    kma_path, gfs_path = _write_inputs(tmp_path)
    result = build_kma_gfs_disagreement_features(kma_path, gfs_path)

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2024-01-01 01:00:00")
    assert result.attrs["issue_times"][0] == pd.Timestamp(
        "2023-12-31 13:00:00"
    )
    row = result.iloc[0]
    assert row["kma_um_u10_mean"] == pytest.approx(4.0)
    assert row["kma_um_speed10_mean"] == pytest.approx(9.0)
    assert row["kma_gfs_delta_u10"] == pytest.approx(2.0)
    assert row["kma_gfs_delta_v10"] == pytest.approx(8.0)
    assert row["kma_gfs_vector_disagreement"] == pytest.approx(np.hypot(2.0, 8.0))
    assert row["kma_gfs_speed_disagreement"] == pytest.approx(7.0)
    assert row["kma_vs_gfs100_speed_ratio"] == pytest.approx(2.25)


def test_build_kma_features_rejects_duplicate_points(tmp_path) -> None:
    kma_path, gfs_path = _write_inputs(tmp_path)
    kma = pd.read_csv(kma_path, encoding="utf-8-sig")
    kma = pd.concat([kma, kma.iloc[[0]]], ignore_index=True)
    kma.to_csv(kma_path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="duplicate"):
        build_kma_gfs_disagreement_features(kma_path, gfs_path)
