import numpy as np
import pandas as pd

from src.diagnostics import detailed_group_metrics, slice_diagnostics


def test_detailed_metrics_include_weighted_hits_and_bias() -> None:
    actual = np.array([100.0, 900.0])
    pred = np.array([100.0, 830.0])

    result = detailed_group_metrics(actual, pred, 1_000.0)

    assert np.isclose(result["hit_rate_6pct"], 0.5)
    assert np.isclose(result["hit_rate_8pct"], 1.0)
    assert np.isclose(result["actual_weighted_hit_rate_6pct"], 0.1)
    assert np.isclose(result["actual_weighted_hit_rate_8pct"], 1.0)
    assert np.isclose(result["bias_kwh"], -35.0)


def test_slice_diagnostics_produces_requested_axes() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=8, freq="h"),
        "y_true": np.linspace(100, 900, 8),
        "y_pred": np.linspace(110, 850, 8),
        "lead_hour": np.arange(12, 20),
        "wind_speed": np.linspace(2, 15, 8),
        "wind_direction": np.arange(0, 360, 45),
        "nwp_disagreement": np.arange(8),
    })

    result = slice_diagnostics(frame, capacity=1_000.0)

    assert "month" in result
    assert "actual_power_bin_diagnostic_only" in result
    assert "lead_time_bucket" in result
    assert "wind_speed_bucket" in result
    assert "wind_direction_sector" in result
    assert "ldaps_gfs_disagreement_quartile" in result
