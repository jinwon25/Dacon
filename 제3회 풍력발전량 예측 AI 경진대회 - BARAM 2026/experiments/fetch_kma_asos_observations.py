"""Fetch causally available KMA ASOS observations for issue-time features.

The competition prediction reference is ``data_available_kst_dtm``.  This
collector never exposes an observation newer than ``reference - safe lag`` and
retains the original provider response, redacted URL, retrieval timestamp and
checksum.  The default station is Taebaek ASOS 216, the nearest long-running
ASOS station to the supplied group-3 coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.compliance import audit_observation_availability


ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
DOCUMENTATION_URL = "https://apihub.kma.go.kr/apiList.do?apiSeq=2"
LICENSE_URL = "https://apihub.kma.go.kr/apiInfo.do"
DEFAULT_STATIONS = (216,)
DEFAULT_WINDOWS = (1, 3, 6, 12, 24)
CORE_COLUMNS = (
    "WD",
    "WS",
    "GST_WD",
    "GST_WS",
    "PA",
    "PS",
    "PT",
    "PR",
    "TA",
    "TD",
    "HM",
    "RN",
    "RN_DAY",
    "RN_INT",
)
# Official KMA ASOS fixed-width schema.  In the provider's first header row,
# grouped names such as GST/RN/SD/CA/CT/TE are repeated and their suffixes are
# printed on a second header row.  Using the published positional schema avoids
# duplicate pandas columns while retaining support for simpler CSV-like fixtures.
ASOS_FIXED_SCHEMA = (
    "TM",
    "STN",
    "WD",
    "WS",
    "GST_WD",
    "GST_WS",
    "GST_TM",
    "PA",
    "PS",
    "PT",
    "PR",
    "TA",
    "TD",
    "HM",
    "PV",
    "RN",
    "RN_DAY",
    "RN_JUN",
    "RN_INT",
    "SD_HR3",
    "SD_DAY",
    "SD_TOT",
    "WC",
    "WP",
    "WW",
    "CA_TOT",
    "CA_MID",
    "CH_MIN",
    "CT",
    "CT_TOP",
    "CT_MID",
    "CT_LOW",
    "VS",
    "SS",
    "SI",
    "ST_GD",
    "TS",
    "TE_005",
    "TE_01",
    "TE_02",
    "TE_03",
    "ST_SEA",
    "WH",
    "BF",
    "IR",
    "IX",
)


@dataclass(frozen=True)
class RequestSpec:
    station_id: int
    start_kst: pd.Timestamp
    end_kst: pd.Timestamp

    @property
    def stem(self) -> str:
        return (
            f"asos_stn{self.station_id:03d}_"
            f"{self.start_kst:%Y%m%d%H}_{self.end_kst:%Y%m%d%H}"
        )


def _kst(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Seoul")
    return timestamp.tz_convert("Asia/Seoul")


def request_specs(
    issue_times: pd.Series | pd.DatetimeIndex,
    station_ids: tuple[int, ...],
    *,
    publication_delay_minutes: int = 120,
    history_hours: int = 24,
    chunk_days: int = 30,
) -> list[RequestSpec]:
    if publication_delay_minutes < 0:
        raise ValueError("publication delay must not be negative")
    if history_hours < 1:
        raise ValueError("history hours must be positive")
    if chunk_days < 1 or chunk_days > 30:
        raise ValueError("ASOS request chunks must contain 1 to 30 days")
    issues = pd.DatetimeIndex(pd.to_datetime(issue_times)).drop_duplicates().sort_values()
    if issues.empty:
        raise ValueError("at least one issue time is required")
    safe = pd.DatetimeIndex(
        [_kst(value) - pd.Timedelta(minutes=publication_delay_minutes) for value in issues]
    )
    first = (safe.min() - pd.Timedelta(hours=history_hours)).floor("h")
    last = safe.max().floor("h")
    specs: list[RequestSpec] = []
    for station_id in sorted(set(int(value) for value in station_ids)):
        cursor = first
        while cursor <= last:
            end = min(cursor + pd.Timedelta(days=chunk_days) - pd.Timedelta(hours=1), last)
            specs.append(RequestSpec(station_id, cursor, end))
            cursor = end + pd.Timedelta(hours=1)
    return specs


def build_url(spec: RequestSpec, api_key: str, *, redact: bool = False) -> str:
    key = "<redacted>" if redact else api_key
    query = urllib.parse.urlencode(
        {
            "tm1": spec.start_kst.strftime("%Y%m%d%H%M"),
            "tm2": spec.end_kst.strftime("%Y%m%d%H%M"),
            "stn": str(spec.station_id),
            "help": "0",
            "authKey": key,
        }
    )
    return f"{ENDPOINT}?{query}"


def _header_tokens(text: str) -> tuple[list[str], int]:
    lines = text.splitlines()
    for position, raw in enumerate(lines):
        tokens = re.split(r"\s+", raw.lstrip("# ").strip())
        upper = [token.upper() for token in tokens]
        if "STN" in upper and "WS" in upper and (
            "TM" in upper or any("YYMMDDHH" in token for token in upper)
        ):
            if (
                len(upper) == len(ASOS_FIXED_SCHEMA)
                and upper[1:4] == ["STN", "WD", "WS"]
            ):
                return list(ASOS_FIXED_SCHEMA), position
            upper[0] = "TM"
            if len(set(upper)) != len(upper):
                raise ValueError(
                    "KMA ASOS response uses an unknown grouped fixed-width schema"
                )
            return upper, position
    raise ValueError("KMA ASOS response does not contain a recognizable field header")


def parse_response(text: str, *, expected_station: int | None = None) -> pd.DataFrame:
    if not text.strip():
        raise ValueError("KMA ASOS API returned an empty response")
    lower = text.lower()
    if any(
        token in lower
        for token in ("invalid auth", "resultcode", "error", "인증키", "오류")
    ):
        raise ValueError("KMA ASOS API returned an authentication or service error")
    header, header_position = _header_tokens(text)
    records: list[list[str]] = []
    for raw in text.splitlines()[header_position + 1 :]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = re.split(r"\s+", stripped)
        if len(tokens) < len(header):
            continue
        records.append(tokens[: len(header)])
    if not records:
        raise ValueError("KMA ASOS response contains no observation rows")
    frame = pd.DataFrame(records, columns=header)
    frame["TM"] = pd.to_datetime(frame["TM"], format="%Y%m%d%H%M", errors="coerce")
    frame["STN"] = pd.to_numeric(frame["STN"], errors="coerce")
    if frame[["TM", "STN"]].isna().any().any():
        raise ValueError("KMA ASOS response has invalid observation keys")
    frame["STN"] = frame["STN"].astype(int)
    if expected_station is not None and set(frame["STN"]) != {int(expected_station)}:
        raise ValueError("KMA ASOS response contains an unexpected station")
    for column in CORE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # KMA uses negative sentinels for fields that cannot physically be negative.
    nonnegative = ("WD", "WS", "GST_WD", "GST_WS", "PA", "PS", "HM", "RN", "RN_DAY", "RN_INT")
    for column in nonnegative:
        frame.loc[frame[column] < 0.0, column] = np.nan
    for column in ("PT", "PR", "TA", "TD"):
        frame.loc[frame[column] <= -90.0, column] = np.nan
    if frame.duplicated(["TM", "STN"]).any():
        raise ValueError("KMA ASOS response contains duplicate station-time rows")
    return frame[["TM", "STN", *CORE_COLUMNS]].sort_values(["TM", "STN"])


def _download(
    spec: RequestSpec,
    api_key: str,
    raw_dir: Path,
    retries: int,
) -> dict[str, Any]:
    output = raw_dir / f"{spec.stem}.txt"
    sidecar = raw_dir / f"{spec.stem}.source.json"
    if output.is_file() and sidecar.is_file():
        retained = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() == retained.get("sha256"):
            parse_response(output.read_text(encoding="utf-8"), expected_station=spec.station_id)
            return retained

    url = build_url(spec, api_key)
    redacted_url = build_url(spec, api_key, redact=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BARAM-Competition-Scientist/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            decoded = payload.decode("utf-8", errors="replace")
            parsed = parse_response(decoded, expected_station=spec.station_id)
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            record = {
                "path": output.as_posix(),
                "source_url": redacted_url,
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "station_id": spec.station_id,
                "observation_start_kst": spec.start_kst.isoformat(),
                "observation_end_kst": spec.end_kst.isoformat(),
                "rows": int(len(parsed)),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            sidecar.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return record
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"KMA ASOS download failed for {spec.stem}: {last_error}")


def load_observations(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for record in raw_records:
        frames.append(
            parse_response(
                Path(record["path"]).read_text(encoding="utf-8"),
                expected_station=int(record["station_id"]),
            )
        )
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["TM", "STN"])
    return result.sort_values(["TM", "STN"]).reset_index(drop=True)


def _slope(values: pd.Series, timestamps: pd.Series) -> float:
    valid = values.notna()
    if int(valid.sum()) < 2:
        return np.nan
    y = values[valid].to_numpy(dtype=float)
    x = (timestamps[valid] - timestamps[valid].min()).dt.total_seconds().to_numpy() / 3600.0
    if np.ptp(x) <= 0.0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_issue_features(
    issue_times: pd.Series | pd.DatetimeIndex,
    observations: pd.DataFrame,
    *,
    publication_delay_minutes: int = 120,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"TM", "STN", *CORE_COLUMNS}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"ASOS observations are missing columns: {sorted(missing)}")
    source = observations.copy()
    source["TM"] = pd.to_datetime(source["TM"])
    if source.duplicated(["TM", "STN"]).any():
        raise ValueError("ASOS observations contain duplicate station-time rows")
    issues = pd.DatetimeIndex(pd.to_datetime(issue_times)).drop_duplicates().sort_values()
    stations = tuple(sorted(int(value) for value in source["STN"].unique()))
    if not stations:
        raise ValueError("ASOS observations contain no stations")

    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    maximum_window = max(windows)
    for issue in issues:
        safe_cutoff = issue - pd.Timedelta(minutes=publication_delay_minutes)
        record: dict[str, Any] = {
            "data_available_kst_dtm": issue,
            "safe_observation_cutoff_kst": safe_cutoff,
        }
        issue_latest: list[pd.Timestamp] = []
        for station in stations:
            station_rows = source[
                (source["STN"] == station)
                & (source["TM"] <= safe_cutoff)
                & (source["TM"] > safe_cutoff - pd.Timedelta(hours=maximum_window))
            ].sort_values("TM")
            prefix = f"asos_stn{station:03d}"
            if station_rows.empty:
                record[f"{prefix}__available"] = 0.0
                continue
            latest = pd.Timestamp(station_rows["TM"].max())
            issue_latest.append(latest)
            record[f"{prefix}__available"] = 1.0
            record[f"{prefix}__latest_age_minutes"] = float(
                (safe_cutoff - latest).total_seconds() / 60.0
            )
            for observed in station_rows["TM"].unique():
                audit_rows.append(
                    {
                        "prediction_reference_kst": issue,
                        "observation_kst": pd.Timestamp(observed),
                        "station_id": station,
                    }
                )
            for hours in windows:
                window = station_rows[
                    station_rows["TM"] > safe_cutoff - pd.Timedelta(hours=hours)
                ].copy()
                name = f"{prefix}__h{hours:02d}"
                record[f"{name}__count"] = float(len(window))
                record[f"{name}__coverage"] = float(len(window) / hours)
                if window.empty:
                    continue
                radians = np.deg2rad(window["WD"].to_numpy(dtype=float))
                speed = window["WS"].to_numpy(dtype=float)
                window["U"] = -speed * np.sin(radians)
                window["V"] = -speed * np.cos(radians)
                for column in ("WS", "U", "V", "GST_WS"):
                    values = window[column]
                    record[f"{name}__{column.lower()}_mean"] = float(values.mean())
                    record[f"{name}__{column.lower()}_std"] = float(values.std(ddof=0))
                    record[f"{name}__{column.lower()}_max"] = float(values.max())
                    record[f"{name}__{column.lower()}_last"] = float(values.iloc[-1])
                    record[f"{name}__{column.lower()}_slope"] = _slope(values, window["TM"])
                mean_speed = float(window["WS"].mean())
                resultant = float(np.hypot(window["U"].mean(), window["V"].mean()))
                record[f"{name}__direction_persistence"] = (
                    resultant / mean_speed if mean_speed > 1e-6 else 0.0
                )
                for column in ("PA", "PS", "TA", "TD", "HM"):
                    values = window[column]
                    record[f"{name}__{column.lower()}_mean"] = float(values.mean())
                    record[f"{name}__{column.lower()}_last"] = float(values.iloc[-1])
                    record[f"{name}__{column.lower()}_slope"] = _slope(values, window["TM"])
                record[f"{name}__rain_sum"] = float(window["RN"].fillna(0.0).sum())
            record[f"{prefix}__latest_observation_kst"] = latest
        if not issue_latest:
            raise ValueError(f"No causal ASOS observations are available for issue {issue}")
        record["latest_observation_kst"] = max(issue_latest)
        records.append(record)

    features = pd.DataFrame.from_records(records).sort_values("data_available_kst_dtm")
    if features["data_available_kst_dtm"].duplicated().any():
        raise ValueError("ASOS issue features contain duplicate issue rows")
    audit = pd.DataFrame.from_records(audit_rows).drop_duplicates()
    audit_observation_availability(
        audit,
        conservative_publication_delay=timedelta(minutes=publication_delay_minutes),
    )
    return features.reset_index(drop=True), audit.reset_index(drop=True)


def _parse_stations(value: str) -> tuple[int, ...]:
    stations = tuple(
        int(token) for token in re.split(r"[:,\s]+", value.strip()) if token.strip()
    )
    if not stations:
        raise argparse.ArgumentTypeError("at least one station id is required")
    return stations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output-dir", default="artifacts_final/external_weather/kma_asos_2024"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-02")
    parser.add_argument("--stations", type=_parse_stations, default=DEFAULT_STATIONS)
    parser.add_argument("--publication-delay-minutes", type=int, default=120)
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--api-key-env", default="KMA_API_KEY")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the exact request scope without requiring an API key",
    )
    args = parser.parse_args()
    metadata = pd.read_csv(
        args.metadata,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    ).drop_duplicates()
    metadata["forecast_kst_dtm"] = pd.to_datetime(metadata["forecast_kst_dtm"])
    metadata["data_available_kst_dtm"] = pd.to_datetime(
        metadata["data_available_kst_dtm"]
    )
    metadata = metadata[
        (metadata["forecast_kst_dtm"] >= pd.Timestamp(args.start))
        & (metadata["forecast_kst_dtm"] < pd.Timestamp(args.end))
    ]
    if metadata.empty:
        raise ValueError("No forecast metadata rows remain after filtering")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    issues = metadata["data_available_kst_dtm"].drop_duplicates()
    specs = request_specs(
        issues,
        args.stations,
        publication_delay_minutes=args.publication_delay_minutes,
        history_hours=args.history_hours,
        chunk_days=args.chunk_days,
    )
    if args.plan_only:
        plan = {
            "provider": "Korea Meteorological Administration",
            "dataset": "KMA ASOS hourly observations",
            "download_started": False,
            "issues": int(len(issues)),
            "stations": list(args.stations),
            "publication_delay_minutes": args.publication_delay_minutes,
            "history_hours": args.history_hours,
            "request_count": int(len(specs)),
            "requests": [
                {
                    "station_id": spec.station_id,
                    "start_kst": spec.start_kst.isoformat(),
                    "end_kst": spec.end_kst.isoformat(),
                    "redacted_url": build_url(spec, "unused", redact=True),
                }
                for spec in specs
            ],
        }
        plan_path = output_dir / "request_plan.json"
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} to a user-issued KMA APIHub key; keys are never "
            "accepted as command-line arguments or written to artifacts"
        )
    raw_records = [_download(spec, api_key, raw_dir, args.retries) for spec in specs]
    observations = load_observations(raw_records)
    features, audit_rows = build_issue_features(
        issues,
        observations,
        publication_delay_minutes=args.publication_delay_minutes,
        windows=tuple(value for value in DEFAULT_WINDOWS if value <= args.history_hours),
    )
    feature_path = output_dir / "issue_features.csv"
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    audit = audit_observation_availability(
        audit_rows,
        conservative_publication_delay=timedelta(
            minutes=args.publication_delay_minutes
        ),
    )
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "Korea Meteorological Administration",
        "dataset": "KMA ASOS hourly observations",
        "source_type": "timestamped_observation_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "KMA public data/API terms; source attribution required",
        "license_url": LICENSE_URL,
        "retrieved_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "coverage": {
            "forecast_metadata": args.metadata,
            "issues": int(len(features)),
            "stations": list(args.stations),
            "history_hours": args.history_hours,
            "feature_columns": int(len(features.columns)),
        },
        "availability_evidence": {
            "method": "observation_timestamp_with_conservative_lag",
            "conservative_delay_minutes": args.publication_delay_minutes,
            "evidence_url": DOCUMENTATION_URL,
        },
        "causality_audit": audit,
        "feature_file": {
            "path": feature_path.as_posix(),
            "rows": int(len(features)),
            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        },
        "raw_files": raw_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
