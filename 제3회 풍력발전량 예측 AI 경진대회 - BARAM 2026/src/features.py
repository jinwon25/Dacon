from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIME_COL = "forecast_kst_dtm"
META_COLS = {TIME_COL, "data_available_kst_dtm", "grid_id", "latitude", "longitude"}

WIND_PAIRS = {
    "ldaps": {
        "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        "ws5_bl": ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
        "ws50_max": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
        "ws50_min": ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
    },
    "gfs": {
        "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        "ws80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
        "ws100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
        "ws_pbl": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
        "ws850": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
        "ws700": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
        "ws500": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
    },
}


def _add_wind_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    for name, (u_col, v_col) in WIND_PAIRS[source].items():
        if u_col in df and v_col in df:
            u = df[u_col].astype("float32")
            v = df[v_col].astype("float32")
            df[name] = np.sqrt(u * u + v * v).astype("float32")
    return df


def _weather_features(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = _add_wind_features(df, source)

    value_cols = [c for c in df.columns if c not in META_COLS]
    df[value_cols] = df[value_cols].astype("float32")

    # Preserve grid-level wind fields. Less relevant variables are retained as spatial
    # summaries to control dimensionality and keep experiment turnaround practical.
    wind_component_cols = {
        col for pair in WIND_PAIRS[source].values() for col in pair
    }
    grid_value_cols = [
        c for c in value_cols
        if c in wind_component_cols or c.startswith("ws") or c == "surface_0_gust"
    ]
    wide = df.pivot(index=TIME_COL, columns="grid_id", values=grid_value_cols)
    wide.columns = [f"{source}__{value}__grid_{grid}" for value, grid in wide.columns]

    # Add compact spatial summaries, which remain robust when individual grid forecasts are missing.
    grouped = df.groupby(TIME_COL, sort=True)[value_cols]
    summary = grouped.agg(["mean", "std", "min", "max"])
    summary.columns = [f"{source}__{value}__{stat}" for value, stat in summary.columns]

    out = wide.join(summary, how="outer").sort_index()
    return out.astype("float32")


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    dayofyear = index.dayofyear.to_numpy()
    data = {
        "hour": hour.astype("int8"),
        "month": index.month.to_numpy(dtype="int8"),
        "dayofweek": index.dayofweek.to_numpy(dtype="int8"),
        # One daily forecast cycle: 01:00 is +12h and the following 00:00 is +35h.
        "lead_hour": (((hour - 1) % 24) + 12).astype("int8"),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dayofyear / 365.25),
    }
    return pd.DataFrame(data, index=index).astype("float32")


def build_features(data_dir: str | Path, split: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    ldaps = _weather_features(data_dir / split / f"ldaps_{split}.csv", "ldaps")
    gfs = _weather_features(data_dir / split / f"gfs_{split}.csv", "gfs")
    features = ldaps.join(gfs, how="inner")
    features = features.join(_calendar_features(features.index))
    features.index.name = TIME_COL
    return features.sort_index()
