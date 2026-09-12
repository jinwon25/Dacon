import numpy as np
import pandas as pd

from src.scada import ScadaGroupSpec, aggregate_scada_hourly, clean_scada_group


def _raw_frame() -> pd.DataFrame:
    times = pd.date_range("2024-01-01 01:00:00", periods=6, freq="10min")
    frame = pd.DataFrame({"kst_dtm": times})
    for turbine_id in (1, 2):
        frame[f"toy_wtg{turbine_id:02d}_power_kw10m"] = 100.0 * turbine_id
        frame[f"toy_wtg{turbine_id:02d}_ws"] = 8.0 + turbine_id
        frame[f"toy_wtg{turbine_id:02d}_wd"] = 350.0 if turbine_id == 1 else 10.0
    return frame


def test_scada_sums_six_intervals_and_aligns_to_hour_end() -> None:
    spec = ScadaGroupSpec("toy.csv", "toy", (1, 2), 1_200.0, 2_400.0)
    hourly = aggregate_scada_hourly(_raw_frame(), spec)
    assert list(hourly.index) == [pd.Timestamp("2024-01-01 02:00:00")]
    assert hourly.iloc[0]["scada_power_kwh"] == 1_800.0
    assert bool(hourly.iloc[0]["is_complete"])
    assert min(hourly.iloc[0]["scada_wind_direction"], 360 - hourly.iloc[0]["scada_wind_direction"]) < 1e-9


def test_scada_flags_and_removes_negative_and_corrupt_power() -> None:
    spec = ScadaGroupSpec("toy.csv", "toy", (1, 2), 1_200.0, 2_400.0)
    raw = _raw_frame()
    raw.loc[0, "toy_wtg01_power_kw10m"] = -999.0
    raw.loc[1, "toy_wtg02_power_kw10m"] = 50_000.0
    interval = clean_scada_group(raw, spec)
    assert interval["negative_sentinel_count"].sum() == 1
    assert interval["corrupt_spike_count"].sum() == 1
    assert np.isnan(interval.loc[0, "interval_power_kwh"])
    hourly = aggregate_scada_hourly(raw, spec)
    assert np.isnan(hourly.iloc[0]["scada_power_kwh"])
    assert not bool(hourly.iloc[0]["is_complete"])
