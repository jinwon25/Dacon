from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_audit import day_ahead_cutoff


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


def _select_latest_legal_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Select the latest forecast cycle available by the day-ahead cutoff."""
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"], errors="raise")
    cutoff = day_ahead_cutoff(df[TIME_COL])
    legal = df.loc[df["data_available_kst_dtm"].to_numpy() <= cutoff].copy()
    pair_cols = [TIME_COL, "grid_id"]
    if legal.groupby(pair_cols).ngroups != df.groupby(pair_cols).ngroups:
        raise ValueError("At least one forecast/grid pair has no cycle available by the 14:00 KST cutoff")
    latest = legal.groupby(pair_cols, sort=False)["data_available_kst_dtm"].transform("max")
    selected = legal.loc[legal["data_available_kst_dtm"] == latest]
    if selected.duplicated(pair_cols).any():
        raise ValueError("Latest legal weather cycle contains duplicate forecast/grid rows")
    return selected


def _weather_features(path: Path, source: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _select_latest_legal_cycle(df)
    availability_counts = df.groupby(TIME_COL)["data_available_kst_dtm"].nunique()
    if not (availability_counts == 1).all():
        raise ValueError(f"{source} has multiple selected availability times for one forecast timestamp")
    availability = df.groupby(TIME_COL, sort=True)["data_available_kst_dtm"].first()
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
    return out.astype("float32"), availability.reindex(out.index)


def _calendar_features(index: pd.DatetimeIndex, availability: pd.Series | None = None) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    dayofyear = index.dayofyear.to_numpy()
    if availability is None:
        # Backward-compatible fallback for callers that construct synthetic
        # indexes. Production build_features always supplies raw availability.
        lead_hour = ((hour - 1) % 24) + 12
    else:
        available_index = pd.DatetimeIndex(pd.to_datetime(availability.reindex(index), errors="raise"))
        lead_hour = (index - available_index).total_seconds() / 3_600.0
        if not np.isfinite(lead_hour).all() or np.any(lead_hour < 0):
            raise ValueError("Invalid lead time derived from data_available_kst_dtm")
    data = {
        "hour": hour.astype("int8"),
        "month": index.month.to_numpy(dtype="int8"),
        "dayofweek": index.dayofweek.to_numpy(dtype="int8"),
        "lead_hour": np.asarray(lead_hour, dtype="float32"),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dayofyear / 365.25),
    }
    return pd.DataFrame(data, index=index).astype("float32")


def _forecast_structure_features(
    index: pd.DatetimeIndex,
    source_availability: dict[str, pd.Series],
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for source, availability in source_availability.items():
        available = pd.DatetimeIndex(pd.to_datetime(availability.reindex(index), errors="raise"))
        data[f"{source}__lead_hour"] = np.asarray((index - available).total_seconds() / 3_600.0, dtype="float32")
        data[f"{source}__forecast_cycle_hour"] = available.hour.astype("float32")
        cutoff = day_ahead_cutoff(index)
        data[f"{source}__cutoff_margin_hour"] = np.asarray((cutoff - available).total_seconds() / 3_600.0, dtype="float32")
    return pd.DataFrame(data, index=index)


def _trajectory_features(
    features: pd.DataFrame,
    availability: pd.Series,
) -> pd.DataFrame:
    """Within-issue NWP trajectory features; every horizon is known at cutoff."""
    candidate_cols = [
        col for col in features.columns
        if col.endswith("__hub_ws117__mean")
        or col.endswith("__ws100__mean")
        or ("__kpx_group_" in col and col.endswith("__hub_ws117__idw"))
        or ("__kpx_group_" in col and col.endswith("__ws100__idw"))
    ]
    issue = pd.Series(pd.to_datetime(availability.reindex(features.index)).to_numpy(), index=features.index)
    output: dict[str, pd.Series] = {}
    for col in candidate_cols:
        grouped = features[col].groupby(issue, sort=False)
        for offset in (-3, -2, -1, 1, 2, 3):
            label = "lag" if offset < 0 else "lead"
            output[f"trajectory__{col}__{label}{abs(offset)}"] = grouped.shift(-offset)
        output[f"trajectory__{col}__ramp1"] = features[col] - grouped.shift(1)
        for stat in ("mean", "std", "min", "max"):
            output[f"trajectory__{col}__issue_{stat}"] = grouped.transform(stat)

    for source in ("ldaps", "gfs"):
        u_col = f"{source}__heightAboveGround_10_10u__mean"
        v_col = f"{source}__heightAboveGround_10_10v__mean"
        if u_col not in features or v_col not in features:
            continue
        direction = pd.Series(np.arctan2(features[v_col], features[u_col]), index=features.index)
        prior = direction.groupby(issue, sort=False).shift(1)
        delta = np.arctan2(np.sin(direction - prior), np.cos(direction - prior))
        output[f"trajectory__{source}__direction_change_1h"] = delta
    return pd.DataFrame(output, index=features.index, dtype="float32")


def _physical_features(features: pd.DataFrame) -> pd.DataFrame:
    output: dict[str, np.ndarray | pd.Series] = {}
    vector_specs = {
        "ldaps_10": ("ldaps__heightAboveGround_10_10u__mean", "ldaps__heightAboveGround_10_10v__mean"),
        "gfs_10": ("gfs__heightAboveGround_10_10u__mean", "gfs__heightAboveGround_10_10v__mean"),
        "gfs_100": ("gfs__heightAboveGround_100_100u__mean", "gfs__heightAboveGround_100_100v__mean"),
    }
    vectors: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for name, (u_col, v_col) in vector_specs.items():
        if u_col not in features or v_col not in features:
            continue
        u = features[u_col].astype(float)
        v = features[v_col].astype(float)
        speed = np.sqrt(u * u + v * v).clip(lower=0.05)
        vectors[name] = (u, v, speed)
        output[f"physical__{name}__vector_speed"] = speed
        output[f"physical__{name}__direction_sin"] = v / speed
        output[f"physical__{name}__direction_cos"] = u / speed

    if "gfs_10" in vectors and "gfs_100" in vectors:
        u10, v10, ws10 = vectors["gfs_10"]
        u100, v100, ws100 = vectors["gfs_100"]
        output["physical__gfs__shear_alpha_10_100"] = np.log(ws100 / ws10) / np.log(10.0)
        dir10 = np.arctan2(v10, u10)
        dir100 = np.arctan2(v100, u100)
        output["physical__gfs__veer_10_100"] = np.arctan2(np.sin(dir100 - dir10), np.cos(dir100 - dir10))
    if "ldaps_10" in vectors and "gfs_10" in vectors:
        lu, lv, lws = vectors["ldaps_10"]
        gu, gv, gws = vectors["gfs_10"]
        output["physical__ldaps_gfs__u_difference"] = lu - gu
        output["physical__ldaps_gfs__v_difference"] = lv - gv
        output["physical__ldaps_gfs__speed_difference"] = lws - gws
        output["physical__ldaps_gfs__vector_disagreement"] = np.sqrt((lu - gu) ** 2 + (lv - gv) ** 2)

    for source, speed_key, temp_col, pressure_col in (
        ("ldaps", "ldaps_10", "ldaps__heightAboveGround_2_t__mean", "ldaps__surface_0_sp__mean"),
        ("gfs", "gfs_100", "gfs__heightAboveGround_2_2t__mean", "gfs__surface_0_sp__mean"),
    ):
        if speed_key not in vectors or temp_col not in features or pressure_col not in features:
            continue
        rho = features[pressure_col].astype(float) / (287.05 * features[temp_col].astype(float).clip(lower=180.0))
        speed = vectors[speed_key][2]
        output[f"physical__{source}__air_density"] = rho
        output[f"physical__{source}__wind_power_density"] = 0.5 * rho * speed ** 3
        output[f"physical__{source}__cut_in_region"] = (speed < 3.5).astype(float)
        output[f"physical__{source}__rated_region"] = ((speed >= 11.0) & (speed < 25.0)).astype(float)
        output[f"physical__{source}__cut_out_region"] = (speed >= 25.0).astype(float)
    return pd.DataFrame(output, index=features.index, dtype="float32")


def build_features(
    data_dir: str | Path,
    split: str,
    feature_blocks: tuple[str, ...] = ("base",),
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    unknown = set(feature_blocks) - {"base", "forecast_structure", "trajectory", "physical"}
    if unknown:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")
    ldaps, ldaps_available = _weather_features(data_dir / split / f"ldaps_{split}.csv", "ldaps")
    gfs, gfs_available = _weather_features(data_dir / split / f"gfs_{split}.csv", "gfs")
    features = ldaps.join(gfs, how="inner")
    ldaps_available = ldaps_available.reindex(features.index)
    gfs_available = gfs_available.reindex(features.index)
    # The generic baseline lead uses the latest source availability. Source-
    # specific leads remain available in the forecast_structure ablation.
    latest_available = pd.concat([ldaps_available, gfs_available], axis=1).max(axis=1)
    features = features.join(_calendar_features(features.index, latest_available))
    if "forecast_structure" in feature_blocks:
        features = features.join(_forecast_structure_features(
            features.index,
            {"ldaps": ldaps_available, "gfs": gfs_available},
        ))
    if "trajectory" in feature_blocks:
        features = features.join(_trajectory_features(features, latest_available))
    if "physical" in feature_blocks:
        features = features.join(_physical_features(features))
    features.index.name = TIME_COL
    return features.sort_index()
