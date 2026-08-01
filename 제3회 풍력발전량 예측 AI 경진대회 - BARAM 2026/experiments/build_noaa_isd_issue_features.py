"""Build causal multi-station issue features from NOAA Global Hourly files.

The downloaded Global Hourly CSVs are observations, not retrospective model
analyses.  Every feature is cut off two hours before the BARAM prediction
reference.  The output intentionally follows the manifest/feature contract of
the existing observation router so the NOAA branch can be compared with the
identical no-observation control without changing the downstream experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "artifacts_final" / "external_weather" / "noaa_isd" / "raw"
DEFAULT_METADATA = (
    ROOT
    / "artifacts_final"
    / "external_weather"
    / "noaa_isd"
    / "metadata"
    / "isd-history.csv"
)
DEFAULT_GROUP3_CACHE = (
    ROOT
    / "artifacts_final"
    / "external_weather"
    / "kma_um_regional_context_2024"
    / "power_curve_oof_20260725.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts_final"
    / "external_weather"
    / "noaa_isd"
    / "issue_features_2024.csv"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_name("manifest_2024.json")
DEFAULT_STATIONS = (
    "47100099999",  # Daegwallyeong, highland
    "47105099999",  # Gangneung
    "47121099999",  # Yeongwol
    "47130099999",  # Uljin
)
HISTORY_HOURS = (1, 3, 6, 12, 24)
CONSERVATIVE_DELAY_MINUTES = 120


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wind(values: pd.Series) -> pd.DataFrame:
    """Decode NOAA WND as meteorological direction and speed in m/s."""
    parts = values.astype("string").str.split(",", expand=True)
    if parts.shape[1] < 5:
        raise ValueError("NOAA WND field does not have five components")
    direction = pd.to_numeric(parts.iloc[:, 0], errors="coerce")
    speed = pd.to_numeric(parts.iloc[:, 3], errors="coerce") / 10.0
    valid = direction.between(0.0, 360.0) & (direction < 999.0)
    valid &= speed.between(0.0, 90.0) & (speed < 99.9)
    radians = np.deg2rad(direction)
    output = pd.DataFrame(
        {
            "wind_direction_deg": direction,
            "wind_speed_ms": speed,
            "wind_u_ms": -speed * np.sin(radians),
            "wind_v_ms": -speed * np.cos(radians),
        },
        index=values.index,
    )
    return output.where(valid)


def load_station_files(paths: list[Path], station: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        raw = pd.read_csv(path, usecols=["STATION", "DATE", "WND"], dtype="string")
        observed_station = raw["STATION"].astype("string").str.zfill(11)
        if not bool((observed_station == station).all()):
            raise ValueError(f"station identity mismatch in {path}")
        timestamp = (
            pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
            .dt.tz_convert("Asia/Seoul")
            .dt.tz_localize(None)
            .dt.floor("h")
        )
        wind = parse_wind(raw["WND"])
        wind.index = pd.DatetimeIndex(timestamp)
        frames.append(wind)
    if not frames:
        raise ValueError(f"no NOAA files supplied for station {station}")
    combined = pd.concat(frames).sort_index()
    combined = combined.loc[~combined.index.isna()]
    combined = combined.astype("float64").groupby(level=0).mean()
    return combined


def _window_features(
    frame: pd.DataFrame,
    issues: pd.DatetimeIndex,
    *,
    delay_minutes: int,
) -> tuple[pd.DataFrame, pd.Series]:
    cutoff = issues - pd.Timedelta(minutes=delay_minutes)
    output = pd.DataFrame(index=issues)
    latest_used = pd.Series(pd.NaT, index=issues, dtype="datetime64[ns]")
    valid_wind = frame.dropna(subset=["wind_speed_ms"])
    if valid_wind.empty:
        return output, latest_used

    positions = valid_wind.index.searchsorted(cutoff, side="right") - 1
    for row, position in enumerate(positions):
        if position < 0:
            continue
        timestamp = valid_wind.index[int(position)]
        latest_used.iloc[row] = timestamp

    for hours in HISTORY_HOURS:
        records: list[dict[str, float]] = []
        for issue, safe_cutoff in zip(issues, cutoff, strict=True):
            start = safe_cutoff - pd.Timedelta(hours=hours)
            window = valid_wind.loc[(valid_wind.index > start) & (valid_wind.index <= safe_cutoff)]
            latest = window.iloc[-1] if not window.empty else None
            records.append(
                {
                    "count": float(len(window)),
                    "ws_mean": float(window["wind_speed_ms"].mean()),
                    "ws_std": float(window["wind_speed_ms"].std(ddof=0)),
                    "ws_max": float(window["wind_speed_ms"].max()),
                    "u_mean": float(window["wind_u_ms"].mean()),
                    "v_mean": float(window["wind_v_ms"].mean()),
                    "u_latest": float(latest["wind_u_ms"]) if latest is not None else np.nan,
                    "v_latest": float(latest["wind_v_ms"]) if latest is not None else np.nan,
                    "ws_latest": float(latest["wind_speed_ms"]) if latest is not None else np.nan,
                }
            )
        block = pd.DataFrame(records, index=issues)
        block["ws_latest_minus_mean"] = block["ws_latest"] - block["ws_mean"]
        block = block.add_prefix(f"h{hours:02d}__")
        output = pd.concat([output, block], axis=1)
    return output, latest_used


def build_issue_features(
    stations: dict[str, pd.DataFrame],
    issues: pd.DatetimeIndex,
    *,
    delay_minutes: int = CONSERVATIVE_DELAY_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    issues = pd.DatetimeIndex(issues).dropna().sort_values().unique()
    feature_blocks: list[pd.DataFrame] = []
    latest_by_station: dict[str, pd.Series] = {}
    for station, frame in stations.items():
        block, latest = _window_features(
            frame,
            issues,
            delay_minutes=delay_minutes,
        )
        block = block.add_prefix(f"asos_noaa_stn{station}__")
        feature_blocks.append(block)
        latest_by_station[station] = latest
    features = pd.concat(feature_blocks, axis=1)

    # Cross-station moments expose terrain disagreement without selecting a
    # station after seeing the validation target.
    pooled: dict[str, pd.Series] = {}
    for hours in HISTORY_HOURS:
        for statistic in (
            "ws_mean",
            "ws_std",
            "ws_max",
            "u_mean",
            "v_mean",
            "u_latest",
            "v_latest",
            "ws_latest",
            "ws_latest_minus_mean",
        ):
            suffix = f"h{hours:02d}__{statistic}"
            columns = [column for column in features if column.endswith(suffix)]
            values = features[columns]
            prefix = f"asos_noaa_pool__h{hours:02d}__{statistic}"
            pooled[prefix + "__mean"] = values.mean(axis=1)
            pooled[prefix + "__std"] = values.std(axis=1, ddof=0)
            pooled[prefix + "__min"] = values.min(axis=1)
            pooled[prefix + "__max"] = values.max(axis=1)
    features = pd.concat([features, pd.DataFrame(pooled, index=issues)], axis=1)

    latest_table = pd.DataFrame(latest_by_station, index=issues)
    latest_observation = latest_table.max(axis=1)
    cutoff = issues - pd.Timedelta(minutes=delay_minutes)
    violations = latest_table.gt(pd.Series(cutoff, index=issues), axis=0)
    if bool(violations.any().any()):
        raise ValueError("NOAA issue features include observations after the safe cutoff")
    margins = (
        pd.Series(issues, index=issues) - latest_observation
    ).dt.total_seconds() / 60.0 - delay_minutes
    finite_margins = margins.dropna()
    audit = {
        "rows": int(latest_table.notna().sum().sum()),
        "violations": 0,
        "reference_column": "prediction_reference_kst",
        "observation_column": "observation_kst",
        "conservative_delay_minutes": int(delay_minutes),
        "minimum_availability_margin_minutes": float(finite_margins.min()),
        "maximum_availability_margin_minutes": float(finite_margins.max()),
        "issues_with_any_observation": int(latest_observation.notna().sum()),
    }
    output = features.copy()
    output.insert(0, "latest_observation_kst", latest_observation)
    output.insert(0, "safe_observation_cutoff_kst", cutoff)
    output.insert(0, "data_available_kst_dtm", issues)
    return output.reset_index(drop=True), audit


def _station_metadata(metadata_path: Path, stations: tuple[str, ...]) -> list[dict[str, Any]]:
    metadata = pd.read_csv(metadata_path, dtype="string")
    station_key = metadata["USAF"].str.zfill(6) + metadata["WBAN"].str.zfill(5)
    selected = metadata.loc[station_key.isin(stations)].copy()
    records = []
    for _, row in selected.iterrows():
        key = str(row["USAF"]).zfill(6) + str(row["WBAN"]).zfill(5)
        records.append(
            {
                "station": key,
                "name": str(row["STATION NAME"]),
                "latitude": float(row["LAT"]),
                "longitude": float(row["LON"]),
                "elevation_m": float(row["ELEV(M)"]),
            }
        )
    if {record["station"] for record in records} != set(stations):
        raise ValueError("NOAA station metadata is incomplete")
    return sorted(records, key=lambda record: record["station"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = _rooted(args.raw_root)
    metadata_path = _rooted(args.metadata)
    group3_cache = _rooted(args.group3_cache)
    output_path = _rooted(args.output)
    manifest_path = _rooted(args.manifest)
    stations = tuple(str(value).zfill(11) for value in args.stations)
    years = tuple(int(value) for value in args.years)

    paths_by_station: dict[str, list[Path]] = {}
    station_frames: dict[str, pd.DataFrame] = {}
    for station in stations:
        paths = [raw_root / str(year) / f"{station}.csv" for year in years]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing NOAA station files: {missing}")
        paths_by_station[station] = paths
        station_frames[station] = load_station_files(paths, station)

    with np.load(group3_cache, allow_pickle=False) as cache:
        issues = pd.DatetimeIndex(pd.to_datetime(cache["issue_ns"]))
    features, audit = build_issue_features(
        station_frames,
        issues,
        delay_minutes=args.delay_minutes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")

    retrieved = datetime.now(timezone.utc).isoformat()
    raw_records: list[dict[str, Any]] = []
    for station, paths in paths_by_station.items():
        for path in paths:
            year = int(path.parent.name)
            raw_records.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "source_url": (
                        "https://www.ncei.noaa.gov/data/global-hourly/"
                        f"access/{year}/{station}.csv"
                    ),
                    "retrieved_at_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "station_id": station,
                    "year": year,
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    raw_records.append(
        {
            "path": metadata_path.relative_to(ROOT).as_posix(),
            "source_url": "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
            "retrieved_at_utc": datetime.fromtimestamp(
                metadata_path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "bytes": int(metadata_path.stat().st_size),
            "sha256": _sha256(metadata_path),
        }
    )
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "NOAA National Centers for Environmental Information",
        "dataset": "Global Hourly / Integrated Surface Database",
        "source_type": "timestamped_observation_archive",
        "documentation_url": (
            "https://www.ncei.noaa.gov/products/land-based-station/"
            "integrated-surface-database"
        ),
        "license": "U.S. government public data; NOAA attribution retained",
        "license_url": "https://www.noaa.gov/disclaimer",
        "retrieved_at_utc": retrieved,
        "coverage": {
            "issues": int(len(features)),
            "stations": _station_metadata(metadata_path, stations),
            "years": list(years),
            "history_hours": max(HISTORY_HOURS),
            "feature_columns": int(
                sum(column.startswith("asos_") for column in features)
            ),
        },
        "availability_evidence": {
            "method": "observation_timestamp_with_conservative_lag",
            "conservative_delay_minutes": int(args.delay_minutes),
            "evidence_url": (
                "https://www.ncei.noaa.gov/products/land-based-station/"
                "integrated-surface-database"
            ),
        },
        "causality_audit": audit,
        "feature_file": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "rows": int(len(features)),
            "sha256": _sha256(output_path),
        },
        "raw_files": raw_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_years(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--group3-cache", default=DEFAULT_GROUP3_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--stations", type=_parse_csv, default=DEFAULT_STATIONS)
    parser.add_argument("--years", type=_parse_years, default=(2023, 2024))
    parser.add_argument(
        "--delay-minutes", type=int, default=CONSERVATIVE_DELAY_MINUTES
    )
    args = parser.parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "feature_file": manifest["feature_file"],
                "coverage": manifest["coverage"],
                "causality_audit": manifest["causality_audit"],
                "manifest": _rooted(args.manifest).relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
