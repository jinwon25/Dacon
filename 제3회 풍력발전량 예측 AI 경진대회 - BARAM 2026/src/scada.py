"""SCADA cleaning and hourly alignment for training-only physical models.

The raw ``*_power_kw10m`` fields are 10-minute energy-like readings.  The
competition label is hourly kWh ending at ``kst_dtm``; consequently six
10-minute readings are summed. VESTAS uses ``ceil(raw_time)`` while UNISON
uses ``floor(raw_time) + 1 hour``. Both are verified against KPX labels in the
P0 cross-correlation audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScadaGroupSpec:
    file_name: str
    prefix: str
    turbine_ids: tuple[int, ...]
    turbine_capacity_kw: float
    group_capacity_kwh: float
    hourly_bucket: str = "floor_plus_1h"


SCADA_GROUP_SPECS: dict[str, ScadaGroupSpec] = {
    "kpx_group_1": ScadaGroupSpec("scada_vestas_train.csv", "vestas", tuple(range(1, 7)), 3_600.0, 21_600.0, "ceil"),
    "kpx_group_2": ScadaGroupSpec("scada_vestas_train.csv", "vestas", tuple(range(7, 13)), 3_600.0, 21_600.0, "ceil"),
    "kpx_group_3": ScadaGroupSpec("scada_unison_train.csv", "unison", tuple(range(1, 6)), 4_200.0, 21_000.0, "floor_plus_1h"),
}


def _columns(spec: ScadaGroupSpec, suffix: str) -> list[str]:
    return [f"{spec.prefix}_wtg{turbine_id:02d}_{suffix}" for turbine_id in spec.turbine_ids]


def clean_scada_group(raw: pd.DataFrame, spec: ScadaGroupSpec) -> pd.DataFrame:
    """Return interval-level values and explicit quality flags.

    Large negative sentinels and absolute values above 10,000 are known
    corrupt spikes. Small negative VESTAS readings are retained as measured
    auxiliary consumption, but explicitly flagged. Mild rated-power
    exceedances are also retained and flagged.
    """

    frame = raw.copy()
    frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"])
    power_cols = _columns(spec, "power_kw10m")
    wind_cols = _columns(spec, "ws")
    direction_cols = _columns(spec, "wd")

    power = frame[power_cols].apply(pd.to_numeric, errors="coerce")
    wind = frame[wind_cols].apply(pd.to_numeric, errors="coerce")
    direction = frame[direction_cols].apply(pd.to_numeric, errors="coerce")

    small_negative = power.lt(0.0) & power.gt(-900.0)
    negative_sentinel = power.le(-900.0)
    corrupt = power.abs().ge(10_000.0)
    interval_rated_kwh = spec.turbine_capacity_kw / 6.0
    above_rated = power.gt(interval_rated_kwh * 1.02) & ~corrupt
    power = power.mask(negative_sentinel | corrupt)
    wind = wind.mask((wind < 0.0) | (wind > 60.0))
    direction = direction.where(direction.notna()) % 360.0

    radians = np.deg2rad(direction)
    out = pd.DataFrame(index=frame.index)
    out["raw_kst_dtm"] = frame["kst_dtm"]
    if spec.hourly_bucket == "ceil":
        out["label_kst_dtm"] = frame["kst_dtm"].dt.ceil("h")
    elif spec.hourly_bucket == "floor_plus_1h":
        out["label_kst_dtm"] = frame["kst_dtm"].dt.floor("h") + pd.Timedelta(hours=1)
    else:
        raise ValueError(f"Unknown SCADA hourly bucket convention: {spec.hourly_bucket}")
    out["interval_power_kwh"] = power.sum(axis=1, min_count=len(power_cols))
    out["mean_wind_speed"] = wind.mean(axis=1)
    out["wind_direction_sin"] = np.sin(radians).mean(axis=1)
    out["wind_direction_cos"] = np.cos(radians).mean(axis=1)
    out["valid_power_turbines"] = power.notna().sum(axis=1)
    out["valid_wind_turbines"] = wind.notna().sum(axis=1)
    out["small_negative_count"] = small_negative.sum(axis=1)
    out["negative_sentinel_count"] = negative_sentinel.sum(axis=1)
    out["corrupt_spike_count"] = corrupt.sum(axis=1)
    out["above_rated_count"] = above_rated.sum(axis=1)
    return out


def aggregate_scada_hourly(raw: pd.DataFrame, spec: ScadaGroupSpec) -> pd.DataFrame:
    """Clean and aggregate one group's 10-minute SCADA to label-ending hours."""

    interval = clean_scada_group(raw, spec)
    grouped = interval.groupby("label_kst_dtm", sort=True)
    hourly = grouped.agg(
        scada_power_kwh=("interval_power_kwh", lambda values: values.sum(min_count=6)),
        scada_wind_speed=("mean_wind_speed", "mean"),
        scada_direction_sin=("wind_direction_sin", "mean"),
        scada_direction_cos=("wind_direction_cos", "mean"),
        interval_count=("raw_kst_dtm", "size"),
        complete_power_intervals=("valid_power_turbines", lambda values: int((values == len(spec.turbine_ids)).sum())),
        small_negative_count=("small_negative_count", "sum"),
        negative_sentinel_count=("negative_sentinel_count", "sum"),
        corrupt_spike_count=("corrupt_spike_count", "sum"),
        above_rated_count=("above_rated_count", "sum"),
    )
    hourly.index.name = "kst_dtm"
    hourly["scada_wind_direction"] = (
        np.rad2deg(np.arctan2(hourly["scada_direction_sin"], hourly["scada_direction_cos"])) % 360.0
    )
    hourly["is_complete"] = (hourly["interval_count"] == 6) & (hourly["complete_power_intervals"] == 6)
    hourly["is_over_capacity"] = hourly["scada_power_kwh"] > spec.group_capacity_kwh * 1.02
    hourly["is_stopped_or_curtailed"] = (
        (hourly["scada_wind_speed"] >= 7.0)
        & (hourly["scada_power_kwh"] <= 0.10 * spec.group_capacity_kwh)
    )
    hourly["is_clean_for_curve"] = (
        hourly["is_complete"]
        & ~hourly["is_over_capacity"]
        & ~hourly["is_stopped_or_curtailed"]
        & hourly["scada_power_kwh"].between(0.0, spec.group_capacity_kwh * 1.02)
        & hourly["scada_wind_speed"].notna()
    )
    return hourly


def load_hourly_scada(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all groups without exposing SCADA to the inference path."""

    train_dir = Path(data_dir) / "train"
    cache: dict[str, pd.DataFrame] = {}
    result: dict[str, pd.DataFrame] = {}
    for target, spec in SCADA_GROUP_SPECS.items():
        if spec.file_name not in cache:
            cache[spec.file_name] = pd.read_csv(train_dir / spec.file_name, encoding="utf-8-sig")
        result[target] = aggregate_scada_hourly(cache[spec.file_name], spec)
    return result
