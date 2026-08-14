from __future__ import annotations

import pandas as pd
import pytest

from experiments.gefs_mean_cycle_disagreement_screen import aggregate_snapshot


def test_snapshot_aggregation_preserves_signed_and_absolute_disagreement() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01 12:00"] * 2),
            "vector_disagreement": [1.0, 3.0],
            "speed_disagreement": [-2.0, 4.0],
            "delta_u10": [-1.0, 2.0],
            "delta_v10": [0.0, 1.0],
            "gefs_speed10": [4.0, 8.0],
            "gfs_speed10": [6.0, 4.0],
        }
    )
    result = aggregate_snapshot(frame)
    assert result.iloc[0]["vector_disagreement_mean"] == pytest.approx(2.0)
    assert result.iloc[0]["speed_disagreement_mean"] == pytest.approx(1.0)
    assert result.iloc[0]["speed_disagreement_abs_mean"] == pytest.approx(3.0)
