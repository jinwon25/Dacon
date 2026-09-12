import numpy as np
import pandas as pd

from experiments.spatiotemporal_multitask import calendar_tensor


def test_spatiotemporal_calendar_uses_actual_availability_for_lead():
    timestamps = pd.date_range("2025-01-01 01:00:00", periods=24, freq="h")
    availability = pd.DatetimeIndex([pd.Timestamp("2024-12-31 13:00:00")] * 24)
    calendar = calendar_tensor(
        timestamps.astype("int64").to_numpy().reshape(1, 24),
        availability.astype("int64").to_numpy().reshape(1, 24),
    )
    np.testing.assert_allclose(calendar[0, :, -1], np.arange(12, 36) / 35.0)


def test_spatiotemporal_calendar_changes_when_issuance_changes():
    timestamps = pd.date_range("2025-01-01 01:00:00", periods=24, freq="h")
    early = pd.DatetimeIndex([pd.Timestamp("2024-12-31 12:00:00")] * 24)
    late = pd.DatetimeIndex([pd.Timestamp("2024-12-31 14:00:00")] * 24)
    early_lead = calendar_tensor(
        timestamps.astype("int64").to_numpy().reshape(1, 24),
        early.astype("int64").to_numpy().reshape(1, 24),
    )[0, 0, -1]
    late_lead = calendar_tensor(
        timestamps.astype("int64").to_numpy().reshape(1, 24),
        late.astype("int64").to_numpy().reshape(1, 24),
    )[0, 0, -1]
    assert np.isclose((early_lead - late_lead) * 35.0, 2.0)
