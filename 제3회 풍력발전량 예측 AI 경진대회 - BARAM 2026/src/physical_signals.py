from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import TIME_COL, TURBINES_BY_GROUP, _distance_weights


REFERENCE_AIR_DENSITY = 1.225
DRY_AIR_GAS_CONSTANT = 287.05


def moist_air_density(
    pressure_pa: pd.Series | np.ndarray,
    temperature_k: pd.Series | np.ndarray,
    specific_humidity: pd.Series | np.ndarray,
) -> np.ndarray:
    """Approximate moist-air density from NWP surface fields.

    The virtual-temperature approximation is sufficiently accurate for a
    model feature and uses only fields available at the forecast issue time.
    """
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    humidity = np.asarray(specific_humidity, dtype=float)
    virtual_temperature = temperature * (1.0 + 0.61 * humidity)
    density = pressure / (DRY_AIR_GAS_CONSTANT * virtual_temperature)
    return np.clip(density, 0.75, 1.45)


def density_normalized_wind_speed(
    wind_speed: pd.Series | np.ndarray,
    air_density: pd.Series | np.ndarray,
    reference_density: float = REFERENCE_AIR_DENSITY,
) -> np.ndarray:
    """IEC-style equivalent wind speed at a constant reference density."""
    speed = np.asarray(wind_speed, dtype=float)
    density = np.asarray(air_density, dtype=float)
    return speed * np.cbrt(density / reference_density)


def hub_height_vector(
    lower_u: pd.Series | np.ndarray,
    lower_v: pd.Series | np.ndarray,
    upper_u: pd.Series | np.ndarray,
    upper_v: pd.Series | np.ndarray,
    lower_height: float,
    upper_height: float,
    hub_height: float = 117.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Power-law extrapolation with the same bounds as the production features."""
    lower_u = np.asarray(lower_u, dtype=float)
    lower_v = np.asarray(lower_v, dtype=float)
    upper_u = np.asarray(upper_u, dtype=float)
    upper_v = np.asarray(upper_v, dtype=float)
    lower_speed = np.maximum(np.hypot(lower_u, lower_v), 0.05)
    upper_speed = np.maximum(np.hypot(upper_u, upper_v), 0.05)
    alpha = np.log(upper_speed / lower_speed) / np.log(upper_height / lower_height)
    alpha = np.nan_to_num(alpha, nan=0.14, posinf=0.14, neginf=0.14)
    alpha = np.clip(alpha, -0.30, 0.60)
    hub_speed = np.clip(upper_speed * (hub_height / upper_height) ** alpha, 0.0, 45.0)
    ratio = hub_speed / upper_speed
    return upper_u * ratio, upper_v * ratio, hub_speed, alpha


def wake_alignment_index(
    eastward_wind: pd.Series | np.ndarray,
    northward_wind: pd.Series | np.ndarray,
    turbines: tuple[tuple[float, float], ...],
    directional_power: int = 8,
) -> np.ndarray:
    """Layout-only proxy for how strongly turbines align with the flow.

    Each unique turbine pair contributes its absolute flow-axis alignment,
    weighted by inverse separation.  This is deliberately a soft feature, not
    a high-fidelity wake-loss calculation.
    """
    u = np.asarray(eastward_wind, dtype=float)
    v = np.asarray(northward_wind, dtype=float)
    speed = np.maximum(np.hypot(u, v), 0.05)
    flow_east = u / speed
    flow_north = v / speed
    pair_scores: list[np.ndarray] = []
    pair_weights: list[float] = []
    for left in range(len(turbines)):
        for right in range(left + 1, len(turbines)):
            lat_1, lon_1 = turbines[left]
            lat_2, lon_2 = turbines[right]
            mean_lat = np.deg2rad((lat_1 + lat_2) / 2.0)
            north_km = (lat_2 - lat_1) * 111.32
            east_km = (lon_2 - lon_1) * 111.32 * np.cos(mean_lat)
            separation = max(float(np.hypot(east_km, north_km)), 0.05)
            alignment = np.abs(
                flow_east * (east_km / separation)
                + flow_north * (north_km / separation)
            )
            pair_scores.append(np.clip(alignment, 0.0, 1.0) ** directional_power)
            pair_weights.append(1.0 / separation)
    if not pair_scores:
        return np.zeros_like(speed)
    return np.average(np.vstack(pair_scores), axis=0, weights=np.asarray(pair_weights))


def _source_columns(source: str) -> list[str]:
    common = [TIME_COL, "grid_id", "latitude", "longitude", "surface_0_sp"]
    if source == "ldaps":
        return common + [
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
            "heightAboveGround_50_50MUmax",
            "heightAboveGround_50_50MUmin",
            "heightAboveGround_50_50MVmax",
            "heightAboveGround_50_50MVmin",
            "heightAboveGround_2_t",
            "heightAboveGround_2_q",
        ]
    if source == "gfs":
        return common + [
            "heightAboveGround_80_u",
            "heightAboveGround_80_v",
            "heightAboveGround_100_100u",
            "heightAboveGround_100_100v",
            "heightAboveGround_2_2t",
            "heightAboveGround_2_2sh",
            "surface_0_gust",
        ]
    raise ValueError(f"Unknown NWP source: {source}")


def _weighted_source_state(path: Path, source: str, target: str) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=_source_columns(source), encoding="utf-8-sig")
    raw[TIME_COL] = pd.to_datetime(raw[TIME_COL])
    if source == "ldaps":
        upper_u = 0.5 * (
            raw["heightAboveGround_50_50MUmax"]
            + raw["heightAboveGround_50_50MUmin"]
        )
        upper_v = 0.5 * (
            raw["heightAboveGround_50_50MVmax"]
            + raw["heightAboveGround_50_50MVmin"]
        )
        hub_u, hub_v, hub_speed, alpha = hub_height_vector(
            raw["heightAboveGround_10_10u"],
            raw["heightAboveGround_10_10v"],
            upper_u,
            upper_v,
            10.0,
            50.0,
        )
        temperature = raw["heightAboveGround_2_t"]
        humidity = raw["heightAboveGround_2_q"]
        gust = None
    else:
        hub_u, hub_v, hub_speed, alpha = hub_height_vector(
            raw["heightAboveGround_80_u"],
            raw["heightAboveGround_80_v"],
            raw["heightAboveGround_100_100u"],
            raw["heightAboveGround_100_100v"],
            80.0,
            100.0,
        )
        temperature = raw["heightAboveGround_2_2t"]
        humidity = raw["heightAboveGround_2_2sh"]
        gust = raw["surface_0_gust"].to_numpy(dtype=float)

    density = moist_air_density(raw["surface_0_sp"], temperature, humidity)
    equivalent_speed = density_normalized_wind_speed(hub_speed, density)
    state = pd.DataFrame(
        {
            TIME_COL: raw[TIME_COL],
            "grid_id": raw["grid_id"],
            "hub_u117": hub_u,
            "hub_v117": hub_v,
            "hub_ws117": hub_speed,
            "shear_alpha": alpha,
            "air_density": density,
            "density_ws117": equivalent_speed,
            "wind_power_density": 0.5 * density * np.asarray(hub_speed) ** 3,
        }
    )
    if gust is not None:
        state["gust_factor"] = gust / np.maximum(np.asarray(hub_speed), 0.5)

    weights = _distance_weights(raw, TURBINES_BY_GROUP[target])
    state["_weight"] = state["grid_id"].map(weights).astype(float)
    value_columns = [c for c in state.columns if c not in {TIME_COL, "grid_id", "_weight"}]
    state[value_columns] = state[value_columns].mul(state["_weight"], axis=0)
    weighted = state.groupby(TIME_COL, sort=True)[value_columns].sum()
    weighted.columns = [f"{source}_{column}" for column in weighted.columns]
    weighted[f"{source}_wake_alignment"] = wake_alignment_index(
        weighted[f"{source}_hub_u117"],
        weighted[f"{source}_hub_v117"],
        TURBINES_BY_GROUP[target],
    )
    return weighted.astype("float32")


def build_group_physical_signals(
    data_dir: str | Path,
    split: str,
    target: str = "kpx_group_3",
) -> pd.DataFrame:
    """Build compact, forecast-time-safe physical/regime features for one group."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if target not in TURBINES_BY_GROUP:
        raise ValueError(f"Unknown target: {target}")
    data_dir = Path(data_dir)
    states = {
        source: _weighted_source_state(
            data_dir / split / f"{source}_{split}.csv", source, target
        )
        for source in ("ldaps", "gfs")
    }
    result = states["ldaps"].join(states["gfs"], how="inner")
    ldaps_speed = result["ldaps_hub_ws117"].astype(float)
    gfs_speed = result["gfs_hub_ws117"].astype(float)
    speed_mean = 0.5 * (ldaps_speed + gfs_speed)
    result["nwp_hub_ws_mean"] = speed_mean
    result["nwp_hub_ws_abs_diff"] = (ldaps_speed - gfs_speed).abs()
    result["nwp_hub_ws_rel_spread"] = (
        result["nwp_hub_ws_abs_diff"] / (speed_mean + 0.5)
    )
    du = result["ldaps_hub_u117"] - result["gfs_hub_u117"]
    dv = result["ldaps_hub_v117"] - result["gfs_hub_v117"]
    result["nwp_hub_vector_diff"] = np.hypot(du, dv)
    dot = (
        result["ldaps_hub_u117"] * result["gfs_hub_u117"]
        + result["ldaps_hub_v117"] * result["gfs_hub_v117"]
    )
    denominator = np.maximum(ldaps_speed * gfs_speed, 0.05)
    result["nwp_direction_agreement"] = np.clip(dot / denominator, -1.0, 1.0)
    result["nwp_density_ws_abs_diff"] = (
        result["ldaps_density_ws117"] - result["gfs_density_ws117"]
    ).abs()
    result["nwp_power_density_log_ratio"] = np.log1p(
        result["ldaps_wind_power_density"]
    ) - np.log1p(result["gfs_wind_power_density"])
    result["nwp_shear_abs_diff"] = (
        result["ldaps_shear_alpha"] - result["gfs_shear_alpha"]
    ).abs()
    result.index.name = TIME_COL
    return result.replace([np.inf, -np.inf], np.nan).astype("float32")
