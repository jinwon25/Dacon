from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIME_COL = "forecast_kst_dtm"
META_COLS = {TIME_COL, "data_available_kst_dtm", "grid_id", "latitude", "longitude"}

TURBINES_BY_GROUP = {
    "kpx_group_1": (
        (37.28211389, 128.95058333),
        (37.28445833, 128.94954167),
        (37.28652500, 128.94971944),
        (37.28975278, 128.95102222),
        (37.29116667, 128.95432778),
        (37.28874444, 128.95693333),
    ),
    "kpx_group_2": (
        (37.28783333, 128.95963056),
        (37.28646944, 128.96312222),
        (37.28360278, 128.96595556),
        (37.28132500, 128.96782778),
        (37.27913611, 128.96697778),
        (37.27516111, 128.96737222),
    ),
    "kpx_group_3": (
        (37.28325833, 128.96249167),
        (37.27789167, 128.97050000),
        (37.27445278, 128.97292778),
        (37.27182778, 128.97472500),
        (37.26856389, 128.97657778),
    ),
}

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
    if source == "ldaps":
        u10 = df["heightAboveGround_10_10u"].astype("float32")
        v10 = df["heightAboveGround_10_10v"].astype("float32")
        u50 = (
            df["heightAboveGround_50_50MUmax"].astype("float32")
            + df["heightAboveGround_50_50MUmin"].astype("float32")
        ) / 2.0
        v50 = (
            df["heightAboveGround_50_50MVmax"].astype("float32")
            + df["heightAboveGround_50_50MVmin"].astype("float32")
        ) / 2.0
        df = _add_hub_height_features(df, u10, v10, u50, v50, lower_height=10.0, upper_height=50.0)
    elif source == "gfs":
        u80 = df["heightAboveGround_80_u"].astype("float32")
        v80 = df["heightAboveGround_80_v"].astype("float32")
        u100 = df["heightAboveGround_100_100u"].astype("float32")
        v100 = df["heightAboveGround_100_100v"].astype("float32")
        df = _add_hub_height_features(df, u80, v80, u100, v100, lower_height=80.0, upper_height=100.0)
    return df


def _add_hub_height_features(
    df: pd.DataFrame,
    lower_u: pd.Series,
    lower_v: pd.Series,
    upper_u: pd.Series,
    upper_v: pd.Series,
    lower_height: float,
    upper_height: float,
    hub_height: float = 117.0,
) -> pd.DataFrame:
    lower_ws = np.sqrt(lower_u * lower_u + lower_v * lower_v).clip(lower=0.05)
    upper_ws = np.sqrt(upper_u * upper_u + upper_v * upper_v).clip(lower=0.05)
    alpha = np.log(upper_ws / lower_ws) / np.log(upper_height / lower_height)
    alpha = alpha.replace([np.inf, -np.inf], 0.14).fillna(0.14).clip(-0.30, 0.60)
    hub_ws = (upper_ws * (hub_height / upper_height) ** alpha).clip(0, 45)
    ratio = (hub_ws / upper_ws).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    hub_u = upper_u * ratio
    hub_v = upper_v * ratio
    df["hub_ws117"] = hub_ws.astype("float32")
    df["hub_u117"] = hub_u.astype("float32")
    df["hub_v117"] = hub_v.astype("float32")
    df["hub_ws117_sq"] = (hub_ws * hub_ws).astype("float32")
    df["hub_ws117_cu"] = (hub_ws * hub_ws * hub_ws).astype("float32")
    df["hub_dir_sin"] = (hub_v / hub_ws.clip(lower=0.05)).clip(-1, 1).astype("float32")
    df["hub_dir_cos"] = (hub_u / hub_ws.clip(lower=0.05)).clip(-1, 1).astype("float32")
    return df


def _distance_weights(df: pd.DataFrame, turbines: tuple[tuple[float, float], ...]) -> dict[int, float]:
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id")
    raw_weights = {}
    for row in grids.itertuples(index=False):
        weight = 0.0
        for lat, lon in turbines:
            mean_lat = np.deg2rad((float(row.latitude) + lat) / 2.0)
            dy = (float(row.latitude) - lat) * 111.32
            dx = (float(row.longitude) - lon) * 111.32 * np.cos(mean_lat)
            distance_km = float(np.sqrt(dx * dx + dy * dy))
            weight += 1.0 / (distance_km + 0.20) ** 2
        raw_weights[int(row.grid_id)] = weight
    total = sum(raw_weights.values())
    return {grid_id: weight / total for grid_id, weight in raw_weights.items()}


def _group_weighted_features(df: pd.DataFrame, source: str, value_cols: list[str]) -> pd.DataFrame:
    selected_cols = [
        c
        for c in value_cols
        if c.startswith(("ws", "hub_"))
        or c.endswith(("_u", "_v", "_10u", "_10v", "_100u", "_100v"))
        or c in {
            "surface_0_gust",
            "heightAboveGround_2_t",
            "heightAboveGround_2_2t",
            "heightAboveGround_2_r",
            "heightAboveGround_2_2r",
            "surface_0_sp",
            "meanSea_0_prmsl",
            "etc_0_blh",
            "planetaryBoundaryLayer_0_VRATE",
        }
    ]
    blocks = []
    for target, turbines in TURBINES_BY_GROUP.items():
        weights = _distance_weights(df, turbines)
        part = df[[TIME_COL, "grid_id", *selected_cols]].copy()
        part["_weight"] = part["grid_id"].map(weights).astype("float32")
        part[selected_cols] = part[selected_cols].mul(part["_weight"], axis=0)
        weighted = part.groupby(TIME_COL, sort=True)[selected_cols].sum()
        weighted.columns = [f"{source}__{target}__{col}__idw" for col in weighted.columns]
        blocks.append(weighted)
    return pd.concat(blocks, axis=1)


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
        if c in wind_component_cols or c.startswith(("ws", "hub_")) or c == "surface_0_gust"
    ]
    wide = df.pivot(index=TIME_COL, columns="grid_id", values=grid_value_cols)
    wide.columns = [f"{source}__{value}__grid_{grid}" for value, grid in wide.columns]

    # Add compact spatial summaries, which remain robust when individual grid forecasts are missing.
    grouped = df.groupby(TIME_COL, sort=True)[value_cols]
    summary = grouped.agg(["mean", "std", "min", "max"])
    summary.columns = [f"{source}__{value}__{stat}" for value, stat in summary.columns]

    weighted = _group_weighted_features(df, source, value_cols)
    out = wide.join(summary, how="outer").join(weighted, how="outer").sort_index()
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
