"""Fetch causal operational previous-run wind forecasts through Open-Meteo.

Open-Meteo's Previous Runs API aligns values to fixed 24/48-hour forecast
offsets.  Every configured model has an explicit conservative publication
delay.  For BARAM leads of 12--35 hours, day-1 values are used only when the
fixed forecast plus that delay precedes the row cutoff; day-2 is used
otherwise.

This is data retrieval, not remote power-model inference. Raw JSON responses,
exact URLs, checksums, and the conservative availability audit are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
DOCUMENTATION_URL = "https://open-meteo.com/en/docs/previous-runs-api"
LICENSE_URL = "https://open-meteo.com/en/licence"
JMA_SCHEDULE_URL = "https://www.jmbsc.or.jp/jp/online/file/f-online10200.html"
JMA_GSM_SCHEDULE_URL = "https://www.jmbsc.or.jp/jp/online/file/f-online10100.html"
ECMWF_DOCUMENTATION_URL = "https://open-meteo.com/en/docs/ecmwf-api"
DWD_DOCUMENTATION_URL = "https://open-meteo.com/en/docs/dwd-api"
GEM_DOCUMENTATION_URL = "https://open-meteo.com/en/docs/gem-api"
PUBLICATION_DELAY_MINUTES = 150
DAY1_MAXIMUM_SAFE_LEAD_HOURS = 21
MODEL_CONFIG = {
    "jma_msm": {
        "label": "JMA MSM",
        "provider": "JMA",
        "publication_delay_minutes": PUBLICATION_DELAY_MINUTES,
        "schedule_url": JMA_SCHEDULE_URL,
    },
    "jma_gsm": {
        "label": "JMA GSM",
        "provider": "JMA",
        # Open-Meteo documents a typical 4--6 hour global-model delay.
        # Six hours is retained as the conservative operational upper bound.
        "publication_delay_minutes": 360,
        "schedule_url": JMA_GSM_SCHEDULE_URL,
    },
    "ecmwf_ifs025": {
        "label": "ECMWF IFS 0.25 degree",
        "provider": "ECMWF",
        # The open-data distribution itself has a documented two-hour
        # additional delay.  Six hours leaves a further four-hour buffer.
        "publication_delay_minutes": 360,
        "schedule_url": ECMWF_DOCUMENTATION_URL,
    },
    "dwd_icon": {
        "label": "DWD ICON Global",
        "provider": "DWD",
        "publication_delay_minutes": 360,
        "schedule_url": DWD_DOCUMENTATION_URL,
    },
    "cmc_gem_gdps": {
        "label": "Environment Canada GEM GDPS",
        "provider": "Environment and Climate Change Canada",
        "publication_delay_minutes": 360,
        "schedule_url": GEM_DOCUMENTATION_URL,
    },
}
VARIABLES = (
    "wind_speed_10m_previous_day1",
    "wind_direction_10m_previous_day1",
    "wind_speed_10m_previous_day2",
    "wind_direction_10m_previous_day2",
)
THERMODYNAMIC_VARIABLES = (
    "temperature_2m_previous_day1",
    "temperature_2m_previous_day2",
    "surface_pressure_previous_day1",
    "surface_pressure_previous_day2",
    "relative_humidity_2m_previous_day1",
    "relative_humidity_2m_previous_day2",
)
DAY3_WIND_VARIABLES = (
    "wind_speed_10m_previous_day3",
    "wind_direction_10m_previous_day3",
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_url(
    latitudes: tuple[float, ...],
    longitudes: tuple[float, ...],
    start_date: str,
    end_date: str,
    *,
    model: str = "jma_msm",
    include_thermodynamics: bool = False,
    include_previous_day3: bool = False,
) -> str:
    if not latitudes or len(latitudes) != len(longitudes):
        raise ValueError("latitude and longitude lists must be non-empty and paired")
    if model not in MODEL_CONFIG:
        raise ValueError(f"unsupported forecast model: {model}")
    variables = VARIABLES + (
        THERMODYNAMIC_VARIABLES if include_thermodynamics else ()
    ) + (
        DAY3_WIND_VARIABLES if include_previous_day3 else ()
    )
    query = {
        "latitude": ",".join(f"{value:.6f}" for value in latitudes),
        "longitude": ",".join(f"{value:.6f}" for value in longitudes),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "models": model,
        "wind_speed_unit": "ms",
        "timezone": "Asia/Seoul",
        "cell_selection": "nearest",
    }
    return f"{API}?{urllib.parse.urlencode(query, safe=',/')}"


def fetch_json(url: str, retries: int = 4) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "baram-competition-scientist/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and parsed.get("error"):
                raise RuntimeError(str(parsed.get("reason", "Open-Meteo error")))
            return payload
        except Exception as exc:  # pragma: no cover - network retry
            error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {error}")


def _wind_components(
    speed: np.ndarray,
    direction_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(direction_degrees)
    # Meteorological direction states where the wind comes from.
    return -speed * np.sin(radians), -speed * np.cos(radians)


def parse_locations(
    payload: bytes,
    *,
    include_thermodynamics: bool = False,
    include_previous_day3: bool = False,
) -> list[pd.DataFrame]:
    raw = json.loads(payload)
    locations = raw if isinstance(raw, list) else [raw]
    variables = VARIABLES + (
        THERMODYNAMIC_VARIABLES if include_thermodynamics else ()
    ) + (
        DAY3_WIND_VARIABLES if include_previous_day3 else ()
    )
    results: list[pd.DataFrame] = []
    for position, location in enumerate(locations, start=1):
        hourly = location.get("hourly", {})
        missing = [name for name in ("time", *variables) if name not in hourly]
        if missing:
            raise ValueError(f"Open-Meteo location {position} is missing: {missing}")
        frame = pd.DataFrame(hourly)
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        if frame["time"].duplicated().any():
            raise ValueError("Open-Meteo response has duplicate hours")
        for variable in variables:
            frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
        day1_columns = [
            "wind_speed_10m_previous_day1",
            "wind_direction_10m_previous_day1",
        ]
        day2_columns = [
            "wind_speed_10m_previous_day2",
            "wind_direction_10m_previous_day2",
        ]
        if frame[day1_columns].isna().any().any():
            raise ValueError("Open-Meteo day-1 wind response is incomplete")
        if include_thermodynamics:
            thermodynamic_day1 = [
                name
                for name in THERMODYNAMIC_VARIABLES
                if name.endswith("previous_day1")
            ]
            if frame[thermodynamic_day1].isna().any().any():
                raise ValueError(
                    "Open-Meteo JMA day-1 thermodynamics are incomplete"
                )
        day2_missing = frame[day2_columns].isna().any(axis=1)
        imputed_day2_rows = int(day2_missing.sum())
        if imputed_day2_rows:
            if imputed_day2_rows > 6:
                raise ValueError("Open-Meteo JMA MSM day-2 gap exceeds six hours")
            day2_speed = frame[day2_columns[0]].to_numpy(dtype=float)
            day2_direction = frame[day2_columns[1]].to_numpy(dtype=float)
            day2_u, day2_v = _wind_components(day2_speed, day2_direction)
            time_index = pd.DatetimeIndex(frame["time"])
            u_series = pd.Series(day2_u, index=time_index).interpolate(
                method="time", limit=6, limit_direction="both"
            )
            v_series = pd.Series(day2_v, index=time_index).interpolate(
                method="time", limit=6, limit_direction="both"
            )
            if u_series.isna().any() or v_series.isna().any():
                raise ValueError("Open-Meteo JMA MSM day-2 gap is not interpolable")
            speed = np.hypot(u_series.to_numpy(), v_series.to_numpy())
            direction = (
                np.rad2deg(
                    np.arctan2(
                        -u_series.to_numpy(),
                        -v_series.to_numpy(),
                    )
                )
                + 360.0
            ) % 360.0
            frame[day2_columns[0]] = speed
            frame[day2_columns[1]] = direction
        interpolated_thermodynamic_rows = 0
        if include_thermodynamics:
            thermodynamic_day2 = [
                name
                for name in THERMODYNAMIC_VARIABLES
                if name.endswith("previous_day2")
            ]
            time_index = pd.DatetimeIndex(frame["time"])
            for variable in thermodynamic_day2:
                missing_rows = int(frame[variable].isna().sum())
                interpolated_thermodynamic_rows = max(
                    interpolated_thermodynamic_rows,
                    missing_rows,
                )
                if missing_rows > 6:
                    raise ValueError(
                        "Open-Meteo JMA day-2 thermodynamic gap "
                        "exceeds six hours"
                    )
                if missing_rows:
                    values = pd.Series(
                        frame[variable].to_numpy(dtype=float),
                        index=time_index,
                    ).interpolate(
                        method="time",
                        limit=6,
                        limit_direction="both",
                    )
                    if values.isna().any():
                        raise ValueError(
                            "Open-Meteo JMA day-2 thermodynamic gap "
                            "is not interpolable"
                        )
                    frame[variable] = values.to_numpy()
        if include_previous_day3 and frame[
            list(DAY3_WIND_VARIABLES)
        ].isna().any().any():
            raise ValueError("Open-Meteo day-3 wind response is incomplete")
        frame.attrs["requested_position"] = position
        frame.attrs["returned_latitude"] = float(location["latitude"])
        frame.attrs["returned_longitude"] = float(location["longitude"])
        frame.attrs["elevation"] = float(location["elevation"])
        frame.attrs["interpolated_day2_rows"] = imputed_day2_rows
        frame.attrs["interpolated_thermodynamic_rows"] = (
            interpolated_thermodynamic_rows
        )
        results.append(frame.set_index("time").sort_index())
    return results


def build_causal_context(
    locations: list[pd.DataFrame],
    primary: pd.DataFrame,
    *,
    compact_stencil: bool = False,
    model: str = "jma_msm",
    publication_delay_minutes: int | None = None,
    include_thermodynamics: bool = False,
    include_previous_day3: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model not in MODEL_CONFIG:
        raise ValueError(f"unsupported forecast model: {model}")
    if publication_delay_minutes is None:
        publication_delay_minutes = int(
            MODEL_CONFIG[model]["publication_delay_minutes"]
        )
    if not 0 < publication_delay_minutes < 24 * 60:
        raise ValueError("publication delay must lie inside one day")
    day1_maximum_safe_lead_hours = int(
        np.floor(24.0 - publication_delay_minutes / 60.0)
    )
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = sorted(required.difference(primary.columns))
    if missing:
        raise ValueError(f"primary context is missing columns: {missing}")
    metadata = primary[list(required)].copy()
    metadata["forecast_kst_dtm"] = pd.to_datetime(
        metadata["forecast_kst_dtm"], errors="raise"
    )
    metadata["data_available_kst_dtm"] = pd.to_datetime(
        metadata["data_available_kst_dtm"], errors="raise"
    )
    if metadata["forecast_kst_dtm"].duplicated().any():
        raise ValueError("primary context contains duplicate forecast timestamps")
    metadata = metadata.set_index("forecast_kst_dtm").sort_index()
    lead_hours = (
        metadata.index.to_series(index=metadata.index)
        - metadata["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    if float(lead_hours.min()) < 12.0 or float(lead_hours.max()) > 35.0:
        raise ValueError("BARAM lead-time contract must remain inside 12--35 hours")
    use_day1 = lead_hours <= day1_maximum_safe_lead_hours
    offset_hours = np.where(use_day1.to_numpy(), 24, 48)
    public_upper_bound = (
        metadata.index
        - pd.to_timedelta(offset_hours, unit="h")
        + pd.Timedelta(minutes=publication_delay_minutes)
    )
    margins = (
        metadata["data_available_kst_dtm"].to_numpy(dtype="datetime64[ns]")
        - public_upper_bound.to_numpy(dtype="datetime64[ns]")
    ) / np.timedelta64(1, "m")
    if np.any(margins < 0):
        raise ValueError(f"{MODEL_CONFIG[model]['label']} policy violates cutoff")

    pieces: list[pd.DataFrame] = []
    location_metadata: list[dict[str, float | int]] = []
    for position, location in enumerate(locations, start=1):
        aligned = location.reindex(metadata.index)
        variables = VARIABLES + (
            THERMODYNAMIC_VARIABLES if include_thermodynamics else ()
        ) + (
            DAY3_WIND_VARIABLES if include_previous_day3 else ()
        )
        if aligned[list(variables)].isna().any().any():
            raise ValueError(
                f"forecast location {position} is incomplete on target hours"
            )
        day1_speed = aligned["wind_speed_10m_previous_day1"].to_numpy(dtype=float)
        day1_direction = aligned[
            "wind_direction_10m_previous_day1"
        ].to_numpy(dtype=float)
        day2_speed = aligned["wind_speed_10m_previous_day2"].to_numpy(dtype=float)
        day2_direction = aligned[
            "wind_direction_10m_previous_day2"
        ].to_numpy(dtype=float)
        day1_u, day1_v = _wind_components(day1_speed, day1_direction)
        day2_u, day2_v = _wind_components(day2_speed, day2_direction)
        selector = use_day1.to_numpy()
        selected_speed = np.where(selector, day1_speed, day2_speed)
        selected_u = np.where(selector, day1_u, day2_u)
        selected_v = np.where(selector, day1_v, day2_v)
        prefix = f"kma_um_ctx_{model}_g{position}"
        part_values: dict[str, np.ndarray] = {
                f"{prefix}__safe_speed10": selected_speed,
                f"{prefix}__safe_u10": selected_u,
                f"{prefix}__safe_v10": selected_v,
                f"{prefix}__day2_speed10": day2_speed,
                f"{prefix}__day2_u10": day2_u,
                f"{prefix}__day2_v10": day2_v,
                f"{prefix}__safe_day1_minus_day2_speed": np.where(
                    selector, day1_speed - day2_speed, 0.0
                ),
                f"{prefix}__safe_day1_minus_day2_u": np.where(
                    selector, day1_u - day2_u, 0.0
                ),
                f"{prefix}__safe_day1_minus_day2_v": np.where(
                    selector, day1_v - day2_v, 0.0
                ),
        }
        if include_previous_day3:
            day3_speed = aligned[
                "wind_speed_10m_previous_day3"
            ].to_numpy(dtype=float)
            day3_direction = aligned[
                "wind_direction_10m_previous_day3"
            ].to_numpy(dtype=float)
            day3_u, day3_v = _wind_components(day3_speed, day3_direction)
            part_values[f"{prefix}__day3_speed10"] = day3_speed
            part_values[f"{prefix}__day3_u10"] = day3_u
            part_values[f"{prefix}__day3_v10"] = day3_v
            part_values[
                f"{prefix}__day2_minus_day3_speed"
            ] = day2_speed - day3_speed
            part_values[f"{prefix}__day2_minus_day3_u"] = day2_u - day3_u
            part_values[f"{prefix}__day2_minus_day3_v"] = day2_v - day3_v
        if include_thermodynamics:
            thermodynamic_pairs = {
                "temperature_2m": (
                    "temperature_2m_previous_day1",
                    "temperature_2m_previous_day2",
                ),
                "surface_pressure": (
                    "surface_pressure_previous_day1",
                    "surface_pressure_previous_day2",
                ),
                "relative_humidity_2m": (
                    "relative_humidity_2m_previous_day1",
                    "relative_humidity_2m_previous_day2",
                ),
            }
            selected_thermo: dict[str, np.ndarray] = {}
            day2_thermo: dict[str, np.ndarray] = {}
            for short_name, (day1_name, day2_name) in (
                thermodynamic_pairs.items()
            ):
                day1_value = aligned[day1_name].to_numpy(dtype=float)
                day2_value = aligned[day2_name].to_numpy(dtype=float)
                selected_value = np.where(selector, day1_value, day2_value)
                selected_thermo[short_name] = selected_value
                day2_thermo[short_name] = day2_value
                part_values[f"{prefix}__safe_{short_name}"] = selected_value
                part_values[f"{prefix}__day2_{short_name}"] = day2_value
                part_values[
                    f"{prefix}__safe_day1_minus_day2_{short_name}"
                ] = np.where(selector, day1_value - day2_value, 0.0)

            def air_density(
                temperature_c: np.ndarray,
                pressure_hpa: np.ndarray,
                relative_humidity: np.ndarray,
            ) -> np.ndarray:
                temperature_k = temperature_c + 273.15
                saturation_hpa = 6.112 * np.exp(
                    17.67 * temperature_c / (temperature_c + 243.5)
                )
                vapor_pa = (
                    saturation_hpa * relative_humidity / 100.0 * 100.0
                )
                dry_pa = pressure_hpa * 100.0 - vapor_pa
                return (
                    dry_pa / (287.05 * temperature_k)
                    + vapor_pa / (461.495 * temperature_k)
                )

            selected_density = air_density(
                selected_thermo["temperature_2m"],
                selected_thermo["surface_pressure"],
                selected_thermo["relative_humidity_2m"],
            )
            day2_density = air_density(
                day2_thermo["temperature_2m"],
                day2_thermo["surface_pressure"],
                day2_thermo["relative_humidity_2m"],
            )
            part_values[f"{prefix}__safe_air_density"] = selected_density
            part_values[f"{prefix}__day2_air_density"] = day2_density
            part_values[
                f"{prefix}__safe_day1_minus_day2_air_density"
            ] = np.where(selector, selected_density - day2_density, 0.0)
            part_values[f"{prefix}__safe_density_speed10"] = (
                selected_speed * np.cbrt(selected_density / 1.225)
            )
        part = pd.DataFrame(part_values, index=metadata.index)
        pieces.append(part)
        location_metadata.append(
            {
                "position": position,
                "returned_latitude": location.attrs["returned_latitude"],
                "returned_longitude": location.attrs["returned_longitude"],
                "elevation": location.attrs["elevation"],
                "interpolated_day2_rows": location.attrs[
                    "interpolated_day2_rows"
                ],
                "interpolated_thermodynamic_rows": location.attrs[
                    "interpolated_thermodynamic_rows"
                ],
            }
        )

    features = pd.concat(pieces, axis=1)
    if compact_stencil:
        if len(locations) != 9:
            raise ValueError("compact forecast stencil requires exactly nine locations")
        suffixes = (
            "safe_speed10",
            "safe_u10",
            "safe_v10",
            "day2_speed10",
            "day2_u10",
            "day2_v10",
            "safe_day1_minus_day2_speed",
            "safe_day1_minus_day2_u",
            "safe_day1_minus_day2_v",
        )
        if include_thermodynamics:
            suffixes += tuple(
                f"{prefix}{variable}"
                for variable in (
                    "temperature_2m",
                    "surface_pressure",
                    "relative_humidity_2m",
                    "air_density",
                )
                for prefix in (
                    "safe_",
                    "day2_",
                    "safe_day1_minus_day2_",
                )
            )
            suffixes += ("safe_density_speed10",)
        if include_previous_day3:
            suffixes += (
                "day3_speed10",
                "day3_u10",
                "day3_v10",
                "day2_minus_day3_speed",
                "day2_minus_day3_u",
                "day2_minus_day3_v",
            )
        compact = pd.DataFrame(index=metadata.index)
        for suffix in suffixes:
            matrix = features[
                [
                    f"kma_um_ctx_{model}_g{position}__{suffix}"
                    for position in range(1, 10)
                ]
            ]
            prefix = f"kma_um_ctx_{model}_stencil__{suffix}"
            compact[f"{prefix}__centre"] = matrix.iloc[:, 4]
            statistics = (
                ("mean", matrix.mean(axis=1)),
                ("std", matrix.std(axis=1, ddof=0)),
            )
            if suffix in {"safe_speed10", "safe_u10", "safe_v10"}:
                statistics += (
                    ("min", matrix.min(axis=1)),
                    ("max", matrix.max(axis=1)),
                )
                compact[f"{prefix}__east_west"] = (
                    matrix.iloc[:, [2, 5, 8]].mean(axis=1)
                    - matrix.iloc[:, [0, 3, 6]].mean(axis=1)
                )
                compact[f"{prefix}__north_south"] = (
                    matrix.iloc[:, [0, 1, 2]].mean(axis=1)
                    - matrix.iloc[:, [6, 7, 8]].mean(axis=1)
                )
            for statistic, values in statistics:
                compact[f"{prefix}__{statistic}"] = values
        features = compact
    features[f"kma_um_ctx_{model}__day1_usable"] = use_day1.astype(float)
    features[f"kma_um_ctx_{model}__selected_offset_hours"] = offset_hours
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("causal forecast features are incomplete")
    result = features.astype("float32")
    result.insert(0, "data_available_kst_dtm", public_upper_bound.to_numpy())
    result.insert(0, "forecast_kst_dtm", result.index)
    audit = {
        "rows": int(len(result)),
        "violations": 0,
        "reference_column": "primary data_available_kst_dtm",
        "availability_column": "conservative model publication upper bound",
        "initialization_semantics": (
            "valid time minus 24h through the model-specific safe lead; "
            "minus 48h thereafter"
        ),
        "model": model,
        "publication_delay_minutes": int(publication_delay_minutes),
        "day1_maximum_safe_lead_hours": int(
            day1_maximum_safe_lead_hours
        ),
        "minimum_availability_margin_minutes": float(np.min(margins)),
        "maximum_availability_margin_minutes": float(np.max(margins)),
        "day1_rows": int(use_day1.sum()),
        "day2_rows": int((~use_day1).sum()),
        "locations": location_metadata,
        "feature_strategy": (
            "compact_3x3_stencil" if compact_stencil else "individual_locations"
        ),
        "thermodynamic_features": bool(include_thermodynamics),
        "previous_day3_wind_features": bool(include_previous_day3),
    }
    return result.reset_index(drop=True), audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model not in MODEL_CONFIG:
        raise ValueError(f"unsupported forecast model: {args.model}")
    model_config = MODEL_CONFIG[args.model]
    primary_path = _rooted(args.primary_context)
    output_dir = _rooted(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(primary_path, encoding="utf-8-sig")
    primary["forecast_kst_dtm"] = pd.to_datetime(
        primary["forecast_kst_dtm"], errors="raise"
    )
    if args.calendar_year is not None:
        primary = primary[
            primary["forecast_kst_dtm"].dt.year == int(args.calendar_year)
        ].copy()
    if args.calendar_start is not None:
        primary = primary[
            primary["forecast_kst_dtm"] >= pd.Timestamp(args.calendar_start)
        ].copy()
    issue_counts = primary.groupby("forecast_kst_dtm")[
        "data_available_kst_dtm"
    ].nunique()
    if not issue_counts.eq(1).all():
        raise ValueError("primary source has conflicting issue times per target")
    primary = primary.drop_duplicates("forecast_kst_dtm").sort_values(
        "forecast_kst_dtm"
    )
    if primary.empty:
        raise ValueError("calendar-year filter selected no primary rows")
    times = pd.to_datetime(primary["forecast_kst_dtm"], errors="raise")
    start_date = times.min().strftime("%Y-%m-%d")
    end_date = times.max().strftime("%Y-%m-%d")
    latitudes = tuple(float(value) for value in args.latitudes.split(","))
    longitudes = tuple(float(value) for value in args.longitudes.split(","))
    url = build_url(
        latitudes,
        longitudes,
        start_date,
        end_date,
        model=args.model,
        include_thermodynamics=bool(args.include_thermodynamics),
        include_previous_day3=bool(args.include_previous_day3),
    )
    raw_path = output_dir / f"open_meteo_{args.model}_previous_runs.json"
    if raw_path.is_file() and not args.force:
        payload = raw_path.read_bytes()
        json.loads(payload)
    else:
        payload = fetch_json(url)
        raw_path.write_bytes(payload)
    locations = parse_locations(
        payload,
        include_thermodynamics=bool(args.include_thermodynamics),
        include_previous_day3=bool(args.include_previous_day3),
    )
    if len(locations) != len(latitudes):
        raise ValueError("Open-Meteo returned an unexpected location count")
    context, causality = build_causal_context(
        locations,
        primary,
        compact_stencil=args.compact_stencil,
        model=args.model,
        publication_delay_minutes=int(
            model_config["publication_delay_minutes"]
        ),
        include_thermodynamics=bool(args.include_thermodynamics),
        include_previous_day3=bool(args.include_previous_day3),
    )
    features_path = output_dir / "features.csv"
    context.to_csv(features_path, index=False, encoding="utf-8-sig")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "artifact_state": "candidate_external_context",
        "raw_files_retained": True,
        "raw_redownload_required": False,
        "provider": (
            f"{model_config['provider']} via Open-Meteo Previous Runs Archive"
        ),
        "dataset": (
            f"{model_config['label']} fixed-lead historical operational "
            "wind forecasts"
        ),
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "upstream_documentation_url": model_config["schedule_url"],
        "license": (
            "Open-Meteo API data under CC BY 4.0; originating-provider "
            "attribution applies"
        ),
        "license_url": LICENSE_URL,
        "retrieved_at_utc": retrieved_at,
        "coverage": {
            "primary_context": primary_path.relative_to(ROOT).as_posix(),
            "calendar_year_filter": args.calendar_year,
            "calendar_start_filter": args.calendar_start,
            "forecast_start_kst": times.min().isoformat(),
            "forecast_end_kst": times.max().isoformat(),
            "targets": int(len(context)),
            "locations": len(locations),
            "feature_strategy": causality["feature_strategy"],
            "model": args.model,
            "variables": list(
                VARIABLES
                + (
                    THERMODYNAMIC_VARIABLES
                    if args.include_thermodynamics
                    else ()
                )
                + (
                    DAY3_WIND_VARIABLES
                    if args.include_previous_day3
                    else ()
                )
            ),
            "fixed_offsets_hours": (
                [24, 48, 72]
                if args.include_previous_day3
                else [24, 48]
            ),
        },
        "availability_evidence": {
            "method": "provider_schedule_with_conservative_lag",
            "conservative_delay_minutes": int(
                model_config["publication_delay_minutes"]
            ),
            "evidence_url": model_config["schedule_url"],
            "fixed_offset_evidence_url": DOCUMENTATION_URL,
        },
        "causality_audit": causality,
        "derived_features": {
            "path": features_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(features_path),
            "rows": int(len(context)),
            "feature_count": int(len(context.columns) - 2),
            "generation_labels_used": False,
            "remote_power_model_inference_used": False,
        },
        "raw_files": [
            {
                "path": raw_path.relative_to(ROOT).as_posix(),
                "source_url": url,
                "retrieved_at_utc": retrieved_at,
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256(raw_path),
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-context", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latitudes", default="37.283")
    parser.add_argument("--longitudes", default="128.950")
    parser.add_argument("--compact-stencil", action="store_true")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIG),
        default="jma_msm",
    )
    parser.add_argument("--include-thermodynamics", action="store_true")
    parser.add_argument("--include-previous-day3", action="store_true")
    parser.add_argument("--calendar-year", type=int)
    parser.add_argument("--calendar-start")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "competition_eligible": report["competition_eligible"],
                "coverage": report["coverage"],
                "causality_audit": report["causality_audit"],
                "derived_features": report["derived_features"],
                "raw_files": report["raw_files"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
