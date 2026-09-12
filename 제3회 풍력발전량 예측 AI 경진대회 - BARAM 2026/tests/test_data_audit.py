import pandas as pd

from src.data_audit import day_ahead_cutoff, prediction_day


def test_midnight_belongs_to_previous_competition_forecast_day() -> None:
    timestamps = pd.to_datetime(["2025-01-01 01:00:00", "2025-01-02 00:00:00", "2025-01-02 01:00:00"])

    days = prediction_day(timestamps)
    cutoffs = day_ahead_cutoff(timestamps)

    assert list(days) == list(pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]))
    assert list(cutoffs) == list(
        pd.to_datetime(["2024-12-31 14:00:00", "2024-12-31 14:00:00", "2025-01-01 14:00:00"])
    )
