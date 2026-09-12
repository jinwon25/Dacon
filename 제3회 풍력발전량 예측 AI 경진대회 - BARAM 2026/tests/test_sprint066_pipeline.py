import numpy as np
import pandas as pd
import pytest

from src.sprint066_pipeline import validate_submission


def _sample() -> pd.DataFrame:
    rows = 8_760
    return pd.DataFrame({
        "forecast_id": [f"forecast_{i:04d}" for i in range(rows)],
        "forecast_kst_dtm": pd.date_range("2025-01-01 01:00:00", periods=rows, freq="h").astype(str),
        "kpx_group_1": np.zeros(rows),
        "kpx_group_2": np.zeros(rows),
        "kpx_group_3": np.zeros(rows),
    })


def test_submission_validator_accepts_exact_sample_contract():
    sample = _sample()
    candidate = sample.copy()
    candidate["kpx_group_1"] = 21_600.0
    candidate["kpx_group_2"] = 1_000.0
    candidate["kpx_group_3"] = 21_000.0
    validate_submission(candidate, sample)


@pytest.mark.parametrize("failure", ["row_count", "timestamp_order", "nan", "capacity"])
def test_submission_validator_rejects_invalid_candidate(failure):
    sample = _sample()
    candidate = sample.copy()
    if failure == "row_count":
        candidate = candidate.iloc[:-1].copy()
    elif failure == "timestamp_order":
        candidate.loc[[0, 1], "forecast_kst_dtm"] = candidate.loc[[1, 0], "forecast_kst_dtm"].to_numpy()
    elif failure == "nan":
        candidate.loc[0, "kpx_group_2"] = np.nan
    elif failure == "capacity":
        candidate.loc[0, "kpx_group_3"] = 21_000.1
    with pytest.raises(ValueError):
        validate_submission(candidate, sample)
