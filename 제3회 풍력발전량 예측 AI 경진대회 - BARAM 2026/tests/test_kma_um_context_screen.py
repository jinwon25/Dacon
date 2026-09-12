from __future__ import annotations

import pandas as pd
import pytest

from experiments.kma_um_context_screen import build_context_screen_features


def test_build_context_screen_features_keeps_numeric_model_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "forecast_kst_dtm": pd.date_range("2024-01-01 01:00", periods=2, freq="h"),
            "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00"),
            "initialization_utc_r0": ["2023-12-30T12:00:00+00:00"] * 2,
            "kma_um_ctx_u10_r0": [1.0, 2.0],
            "kma_um_ctx_constant": [3.0, 3.0],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    result = build_context_screen_features(path)
    assert list(result.columns) == ["kma_um_ctx_u10_r0"]
    assert result.index[0] == pd.Timestamp("2024-01-01 01:00")
    assert result.attrs["issue_times"][0] == pd.Timestamp("2023-12-31 13:00")


def test_build_context_screen_features_rejects_duplicate_issue_rows(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00"] * 2,
            "data_available_kst_dtm": ["2023-12-31 13:00"] * 2,
            "kma_um_ctx_u10_r0": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        build_context_screen_features(path)
