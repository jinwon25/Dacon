from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DOCUMENTATION_URL = "https://open-meteo.com/en/docs/previous-runs-api"
LICENSE_URL = "https://open-meteo.com/en/license"
VARIABLES = (
    "wind_speed_10m_previous_day2",
    "wind_direction_10m_previous_day2",
    "wind_speed_100m_previous_day2",
    "wind_direction_100m_previous_day2",
    "temperature_2m_previous_day2",
    "surface_pressure_previous_day2",
)


def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    output = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=days - 1))
        output.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return output


def _request(params: dict[str, str], retries: int = 3) -> dict[str, object]:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "baram-competition-scientist/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network-dependent retry
            error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {error}")


def fetch(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    model: str,
    chunk_days: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames = []
    requests = []
    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(VARIABLES),
            "models": model,
            "timezone": "Asia/Seoul",
        }
        payload = _request(params)
        hourly = payload.get("hourly", {})
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise ValueError(f"Malformed Open-Meteo response for {chunk_start}")
        frame = pd.DataFrame(hourly)
        frame["forecast_kst_dtm"] = pd.to_datetime(frame.pop("time"))
        frames.append(frame)
        requests.append(
            {
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "resolved_latitude": payload.get("latitude"),
                "resolved_longitude": payload.get("longitude"),
                "elevation_m": payload.get("elevation"),
                "generationtime_ms": payload.get("generationtime_ms"),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates("forecast_kst_dtm").sort_values("forecast_kst_dtm")
    expected = pd.date_range(start, end + timedelta(days=1), freq="h", inclusive="left")
    if not pd.DatetimeIndex(data["forecast_kst_dtm"]).equals(expected):
        raise ValueError("Downloaded timestamps are not a complete hourly KST range")

    for height in (10, 100):
        speed = data[f"wind_speed_{height}m_previous_day2"].to_numpy(float) / 3.6
        direction = np.deg2rad(
            data[f"wind_direction_{height}m_previous_day2"].to_numpy(float)
        )
        data[f"wind_u_{height}m_previous_day2"] = -speed * np.sin(direction)
        data[f"wind_v_{height}m_previous_day2"] = -speed * np.cos(direction)
        data[f"wind_speed_{height}m_previous_day2"] = speed
    return data, requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latitude", type=float, default=37.28)
    parser.add_argument("--longitude", type=float, default=128.96)
    parser.add_argument("--start-date", default="2024-03-15")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--model", default="ecmwf_ifs025")
    parser.add_argument("--chunk-days", type=int, default=60)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/external_weather/"
            "open_meteo_ecmwf_ifs025_previous_day2.csv"
        ),
    )
    parser.add_argument(
        "--research-only-unverified",
        action="store_true",
        help=(
            "Allow a non-submission research download. previous_day offsets alone "
            "do not prove the original public-availability timestamp."
        ),
    )
    args = parser.parse_args()

    if not args.research_only_unverified:
        raise RuntimeError(
            "Blocked for competition use: a previous_day offset is not sufficient "
            "proof of public availability under the BARAM rules. Use a timestamped "
            "operational archive and an eligible external-data manifest instead."
        )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    if not 1 <= args.chunk_days <= 90:
        raise ValueError("chunk-days must be between 1 and 90")

    frame, requests = fetch(
        args.latitude,
        args.longitude,
        start,
        end,
        args.model,
        args.chunk_days,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "competition_eligible": False,
        "ineligibility_reason": (
            "The API offset does not by itself establish the original run's "
            "creation and public-availability timestamp."
        ),
        "source": "Open-Meteo Previous Runs API",
        "source_type": "retrospective_previous_run_api",
        "api_url": API_URL,
        "documentation_url": DOCUMENTATION_URL,
        "license_url": LICENSE_URL,
        "license": "CC BY 4.0",
        "model": args.model,
        "requested_coordinates": {
            "latitude": args.latitude,
            "longitude": args.longitude,
        },
        "period": {"start": args.start_date, "end": args.end_date},
        "timezone": "Asia/Seoul",
        "forecast_age_hours": 48,
        "competition_issue_time_kst": "previous day 13:00",
        "minimum_hours_available_before_issue": 14,
        "leakage_note": "Research-only output; must not be used in a submission.",
        "rows": int(len(frame)),
        "nonnull_counts": {
            column: int(frame[column].notna().sum()) for column in frame.columns
        },
        "requests": requests,
        "output": str(output),
        "sha256": digest,
    }
    report_path = output.with_suffix(".provenance.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
