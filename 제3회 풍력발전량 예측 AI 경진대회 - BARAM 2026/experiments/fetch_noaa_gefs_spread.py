from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agent_service.compliance import (
    audit_forecast_availability,
    latest_safe_forecast_run,
)


BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
DOCUMENTATION_URL = (
    "https://www.ncei.noaa.gov/products/weather-climate-models/"
    "global-ensemble-forecast"
)
LICENSE_URL = "https://www.noaa.gov/disclaimer"
PRODUCT = "gespr"
GRID = "0p50"
SUBDIRECTORY = "pgrb2ap5"
DELAY = timedelta(hours=6, minutes=10)
WIND_PATTERNS = (":UGRD:10 m above ground:", ":VGRD:10 m above ground:")


def object_url(
    initialization_utc: pd.Timestamp,
    lead_hour: int,
    suffix: str = "",
    product: str = PRODUCT,
) -> str:
    date = initialization_utc.strftime("%Y%m%d")
    cycle = initialization_utc.strftime("%H")
    if product not in {"geavg", "gespr"}:
        raise ValueError("product must be geavg or gespr")
    name = f"{product}.t{cycle}z.pgrb2a.{GRID}.f{lead_hour:03d}{suffix}"
    return f"{BUCKET}/gefs.{date}/{cycle}/atmos/{SUBDIRECTORY}/{name}"


def parse_idx_ranges(text: str, object_size: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        entries.append({"offset": offset, "description": ":" + parts[2]})
    if not entries:
        raise ValueError("GEFS index contains no byte offsets")
    entries.sort(key=lambda item: item["offset"])
    for index, entry in enumerate(entries):
        next_offset = (
            entries[index + 1]["offset"] if index + 1 < len(entries) else object_size
        )
        entry["end"] = int(next_offset) - 1
    selected = [
        entry
        for entry in entries
        if any(pattern in entry["description"] for pattern in WIND_PATTERNS)
    ]
    if len(selected) != 2:
        raise ValueError("GEFS index must contain exactly UGRD/VGRD 10 m fields")
    return selected


def build_plans(
    forecast_frame: pd.DataFrame,
    max_issues: int | None = None,
    start_issue: str | None = None,
    end_issue: str | None = None,
    product: str = PRODUCT,
    requested_leads: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = sorted(required.difference(forecast_frame.columns))
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {missing}")
    frame = forecast_frame[list(required)].drop_duplicates().copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    if start_issue:
        frame = frame[frame["data_available_kst_dtm"] >= pd.Timestamp(start_issue)]
    if end_issue:
        frame = frame[frame["data_available_kst_dtm"] <= pd.Timestamp(end_issue)]
    if frame.empty:
        raise ValueError("issue filters selected no forecast rows")
    issues = sorted(frame["data_available_kst_dtm"].unique())
    if max_issues is not None:
        if max_issues < 1:
            raise ValueError("max_issues must be positive")
        issues = issues[:max_issues]
        frame = frame[frame["data_available_kst_dtm"].isin(issues)]

    join_rows: list[dict[str, Any]] = []
    request_rows: dict[tuple[pd.Timestamp, int], dict[str, Any]] = {}
    for issue, block in frame.groupby("data_available_kst_dtm", sort=True):
        selected = latest_safe_forecast_run(issue, conservative_publication_delay=DELAY)
        initialization = selected.initialization_utc
        publication = selected.conservative_publication_utc
        for valid_kst in sorted(block["forecast_kst_dtm"].unique()):
            valid_utc = pd.Timestamp(valid_kst, tz="Asia/Seoul").tz_convert("UTC")
            raw_lead = (valid_utc - initialization).total_seconds() / 3600.0
            lower = int(math.floor(raw_lead / 3.0) * 3)
            upper = int(math.ceil(raw_lead / 3.0) * 3)
            if lower < 0 or upper > 384:
                raise ValueError(f"GEFS lead is outside supported range: {raw_lead}")
            join_rows.append(
                {
                    "forecast_kst_dtm": pd.Timestamp(valid_kst),
                    "data_available_kst_dtm": pd.Timestamp(issue),
                    "initialization_utc": initialization,
                    "public_availability_utc": publication,
                    "raw_lead_hours": raw_lead,
                    "lower_lead_hour": lower,
                    "upper_lead_hour": upper,
                    "upper_weight": (raw_lead - lower) / max(upper - lower, 1),
                }
            )
            for lead in {lower, upper}:
                request_rows[(initialization, lead)] = {
                    "prediction_reference_kst": selected.prediction_reference_kst,
                    "initialization_utc": initialization,
                    "public_availability_utc": publication,
                    "lead_hour": lead,
                    "valid_time_utc": initialization + pd.Timedelta(hours=lead),
                    "grib_url": object_url(initialization, lead, product=product),
                    "idx_url": object_url(initialization, lead, ".idx", product=product),
                }
    joins = pd.DataFrame(join_rows).sort_values("forecast_kst_dtm").reset_index(drop=True)
    requests = pd.DataFrame(request_rows.values()).sort_values(
        ["initialization_utc", "lead_hour"]
    ).reset_index(drop=True)
    if requested_leads is not None:
        leads = sorted({int(value) for value in requested_leads})
        if not leads or any(value < 0 or value > 384 or value % 3 for value in leads):
            raise ValueError("requested leads must be 3-hour increments from 0 to 384")
        requests = requests[requests["lead_hour"].isin(leads)].reset_index(drop=True)
        expected = len(issues) * len(leads)
        if len(requests) != expected:
            raise ValueError("one or more requested leads is outside the target horizon")
    return joins, requests


def _open(request: urllib.request.Request, timeout: int, retries: int = 4):
    error: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except Exception as exc:  # pragma: no cover - network retry
            error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"GEFS request failed after {retries} attempts: {error}")


def _head(url: str) -> tuple[int, pd.Timestamp]:
    request = urllib.request.Request(url, method="HEAD")
    with _open(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        modified = pd.Timestamp(parsedate_to_datetime(response.headers["Last-Modified"]))
    return size, modified.tz_convert("UTC")


def _read_text(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "baram-competition-scientist/1.0"}
    )
    with _open(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _read_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "baram-competition-scientist/1.0",
        },
    )
    with _open(request, timeout=120) as response:
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise ValueError(f"range response has {len(payload)} bytes, expected {expected}")
    return payload


def _replace_with_retry(temporary: Path, destination: Path, retries: int = 6) -> None:
    error: PermissionError | None = None
    for attempt in range(retries):
        try:
            temporary.replace(destination)
            return
        except PermissionError as exc:  # pragma: no cover - Windows file scanner race
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.1 * (attempt + 1))
    raise PermissionError(
        f"could not atomically replace {destination} after {retries} attempts"
    ) from error


def download_requests(
    requests: pd.DataFrame,
    raw_dir: Path,
    workers: int = 8,
    product: str = PRODUCT,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    def download_one(row: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        initialization = pd.Timestamp(row.initialization_utc)
        name = f"{product}_{initialization:%Y%m%d%H}_f{int(row.lead_hour):03d}_uv10.grib2"
        path = raw_dir / name
        sidecar = path.with_suffix(".source.json")
        if path.is_file() and sidecar.is_file():
            try:
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
                raw_item = cached["raw_file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() == raw_item["sha256"]:
                    return cached["audit_row"], raw_item
            except (OSError, KeyError, json.JSONDecodeError):
                # A prior interruption may leave a partial sidecar. The GRIB is
                # not trusted without valid metadata and is fetched again.
                pass

        object_size, last_modified = _head(row.grib_url)
        idx_text = _read_text(row.idx_url)
        ranges = parse_idx_ranges(idx_text, object_size)
        payload = b"".join(
            _read_range(row.grib_url, item["offset"], item["end"])
            for item in ranges
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        _replace_with_retry(temporary, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        audit_row = {
            "prediction_reference_kst": pd.Timestamp(row.prediction_reference_kst).isoformat(),
            "initialization_utc": initialization.isoformat(),
            "public_availability_utc": last_modified.isoformat(),
        }
        raw_item = {
            "path": path.as_posix(),
            "source_url": row.grib_url,
            "index_url": row.idx_url,
            "source_last_modified_utc": last_modified.isoformat(),
            "retrieved_at_utc": retrieved_at,
            "byte_ranges": [
                {
                    "start": item["offset"],
                    "end": item["end"],
                    "description": item["description"],
                }
                for item in ranges
            ],
            "bytes": len(payload),
            "sha256": digest,
        }
        sidecar_temporary = sidecar.with_suffix(".tmp")
        sidecar_temporary.write_text(
            json.dumps(
                {"audit_row": audit_row, "raw_file": raw_item},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _replace_with_retry(sidecar_temporary, sidecar)
        return audit_row, raw_item

    audited_rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    rows = list(requests.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            audit_row, raw_item = future.result()
            audited_rows.append(audit_row)
            raw_files.append(raw_item)
            if completed % 100 == 0 or completed == len(rows):
                print(f"downloaded_or_verified={completed}/{len(rows)}", flush=True)
    raw_files.sort(key=lambda item: item["path"])
    audit_frame = pd.DataFrame(audited_rows)
    return audit_frame, raw_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-metadata", default="data/train/gfs_train.csv")
    parser.add_argument("--output-dir", default="artifacts_final/external_weather/noaa_gefs")
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--start-issue")
    parser.add_argument("--end-issue")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--product", choices=("geavg", "gespr"), default=PRODUCT)
    parser.add_argument(
        "--lead-hours",
        help="Optional comma-separated 3-hour leads; default uses every target bracket.",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(
        args.forecast_metadata,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    requested_leads = (
        tuple(int(value) for value in args.lead_hours.split(",") if value.strip())
        if args.lead_hours
        else None
    )
    joins, requests = build_plans(
        metadata,
        args.max_issues,
        start_issue=args.start_issue,
        end_issue=args.end_issue,
        product=args.product,
        requested_leads=requested_leads,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joins.to_csv(output_dir / "join_plan.csv", index=False, encoding="utf-8-sig")
    requests.to_csv(output_dir / "request_plan.csv", index=False, encoding="utf-8-sig")

    raw_files: list[dict[str, Any]] = []
    causality = {
        "rows": 0,
        "violations": 0,
        "minimum_availability_margin_minutes": -1,
    }
    if args.download:
        audit_frame, raw_files = download_requests(
            requests,
            output_dir / "raw",
            workers=args.workers,
            product=args.product,
        )
        causality = audit_forecast_availability(audit_frame)
        project_root = Path.cwd().resolve()
        for item in raw_files:
            item["path"] = str(Path(item["path"]).resolve().relative_to(project_root)).replace(
                "\\", "/"
            )

    retrieved_at = datetime.now(timezone.utc).isoformat()
    is_mean = args.product == "geavg"
    statistic = "ensemble mean" if is_mean else "ensemble std dev"
    manifest = {
        "schema_version": 1,
        "competition_eligible": bool(args.download and raw_files),
        "provider": "NOAA/NCEP",
        "dataset": f"GEFS operational {statistic} forecast",
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "US government public data; NOAA disclaimer applies",
        "license_url": LICENSE_URL,
        "retrieved_at_utc": retrieved_at,
        "coverage": {
            "forecast_metadata": args.forecast_metadata,
            "issues": int(joins["data_available_kst_dtm"].nunique()),
            "targets": int(len(joins)),
            "objects": int(len(requests)),
            "product": args.product,
            "grid": GRID,
            "variables": [
                f"UGRD:10 m {statistic}",
                f"VGRD:10 m {statistic}",
            ],
        },
        "availability_evidence": {
            "method": "archive_object_timestamp_with_conservative_lag",
            "conservative_delay_minutes": int(DELAY.total_seconds() // 60),
            "evidence_url": BUCKET,
        },
        "causality_audit": causality,
        "raw_files": raw_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
