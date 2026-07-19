from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

import pandas as pd

from agent_service.compliance import (
    audit_forecast_availability,
    latest_safe_forecast_run,
)
from experiments.fetch_noaa_gefs_spread import (
    _read_range,
    _read_text,
    _replace_with_retry,
)


BASE = (
    "https://www.ncei.noaa.gov/data/climate-forecast-system/access/"
    "operational-9-month-forecast/time-series"
)
DOCUMENTATION_URL = (
    "https://www.ncei.noaa.gov/products/weather-climate-models/"
    "climate-forecast-system"
)
PRODUCT_URL = "https://www.nco.ncep.noaa.gov/pmb/products/cfs/"
LICENSE_URL = "https://www.noaa.gov/disclaimer"
DELAY = timedelta(hours=30)
MEMBER = "01"
INVENTORY_ROW = re.compile(
    r"^(?P<message>\d+):(?P<offset>\d+):d=(?P<cycle>\d{10}):"
    r"(?P<component>UGRD|VGRD):10 m above ground:"
    r"(?P<lead>\d+) hour fcst:"
)


def _head_archive(url: str) -> tuple[int, pd.Timestamp]:
    """HEAD an NCEI object without retrying permanent 404 gaps."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as response:
                size = int(response.headers["Content-Length"])
                modified = pd.Timestamp(
                    parsedate_to_datetime(response.headers["Last-Modified"])
                ).tz_convert("UTC")
            return size, modified
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(url) from error
            last_error = error
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"CFSv2 HEAD failed after retries: {url}: {last_error}")


def object_url(initialization_utc: pd.Timestamp, suffix: str = "grb2") -> str:
    initialization = pd.Timestamp(initialization_utc)
    if initialization.tzinfo is None:
        initialization = initialization.tz_localize("UTC")
    initialization = initialization.tz_convert("UTC")
    cycle = initialization.strftime("%Y%m%d%H")
    directory = initialization.strftime("%Y/%Y%m/%Y%m%d/%Y%m%d%H")
    name = f"wnd10m.{MEMBER}.{cycle}.daily.{suffix}"
    return f"{BASE}/{directory}/{name}"


def parse_inventory_ranges(
    text: str, object_size: int, requested_leads: tuple[int, ...]
) -> list[dict[str, Any]]:
    requested = {int(value) for value in requested_leads}
    if not requested or any(value <= 0 or value % 6 for value in requested):
        raise ValueError("CFSv2 requested leads must be positive 6-hour increments")
    offsets: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        offsets.append({"offset": offset, "line": line})
    if not offsets:
        raise ValueError("CFSv2 inventory contains no byte offsets")
    offsets.sort(key=lambda item: item["offset"])
    for position, item in enumerate(offsets):
        next_offset = (
            offsets[position + 1]["offset"]
            if position + 1 < len(offsets)
            else object_size
        )
        item["end"] = int(next_offset) - 1

    selected = []
    for item in offsets:
        match = INVENTORY_ROW.match(item["line"])
        if match is None:
            continue
        lead = int(match.group("lead"))
        if lead not in requested:
            continue
        selected.append(
            {
                "offset": item["offset"],
                "end": item["end"],
                "component": match.group("component"),
                "lead_hour": lead,
                "description": item["line"],
            }
        )
    expected = {(component, lead) for component in ("UGRD", "VGRD") for lead in requested}
    actual = {(item["component"], item["lead_hour"]) for item in selected}
    if actual != expected or len(selected) != len(expected):
        missing = sorted(expected.difference(actual))
        raise ValueError(f"CFSv2 inventory is missing requested U/V messages: {missing}")
    return sorted(selected, key=lambda item: item["offset"])


def build_plans(
    forecast_frame: pd.DataFrame,
    max_issues: int | None = None,
    start_issue: str | None = None,
    end_issue: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = sorted(required.difference(forecast_frame.columns))
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {missing}")
    frame = forecast_frame[list(required)].drop_duplicates().copy()
    for column in required:
        frame[column] = pd.to_datetime(frame[column])
    if start_issue:
        frame = frame[frame["data_available_kst_dtm"] >= pd.Timestamp(start_issue)]
    if end_issue:
        frame = frame[frame["data_available_kst_dtm"] <= pd.Timestamp(end_issue)]
    if frame.empty:
        raise ValueError("issue filters selected no forecast rows")
    issues = frame["data_available_kst_dtm"].drop_duplicates().sort_values()
    if max_issues is not None:
        if max_issues < 1:
            raise ValueError("max_issues must be positive")
        issues = issues.head(max_issues)
        frame = frame[frame["data_available_kst_dtm"].isin(issues)]

    joins: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for issue, block in frame.groupby("data_available_kst_dtm", sort=True):
        safe = latest_safe_forecast_run(
            issue,
            conservative_publication_delay=DELAY,
            lookback_days=4,
        )
        initialization = safe.initialization_utc
        query_leads: set[int] = set()
        for valid_kst in sorted(block["forecast_kst_dtm"].unique()):
            valid_utc = pd.Timestamp(valid_kst, tz="Asia/Seoul").tz_convert("UTC")
            raw_lead = (valid_utc - initialization).total_seconds() / 3600.0
            lower = int(math.floor(raw_lead / 6.0) * 6)
            upper = int(math.ceil(raw_lead / 6.0) * 6)
            if lower <= 0 or upper > 216:
                raise ValueError(f"CFSv2 target lead is outside pilot range: {raw_lead}")
            query_leads.update((lower, upper))
            joins.append(
                {
                    "forecast_kst_dtm": pd.Timestamp(valid_kst),
                    "data_available_kst_dtm": pd.Timestamp(issue),
                    "initialization_utc": initialization,
                    "public_availability_utc": safe.conservative_publication_utc,
                    "raw_lead_hours": raw_lead,
                    "lower_lead_hour": lower,
                    "upper_lead_hour": upper,
                    "upper_weight": (raw_lead - lower) / max(upper - lower, 1),
                }
            )
        requests.append(
            {
                "prediction_reference_kst": safe.prediction_reference_kst,
                "initialization_utc": initialization,
                "public_availability_utc": safe.conservative_publication_utc,
                "lead_hours": ",".join(str(value) for value in sorted(query_leads)),
                "grib_url": object_url(initialization),
                "inventory_url": object_url(initialization, "inv"),
            }
        )
    join_frame = pd.DataFrame(joins).sort_values("forecast_kst_dtm").reset_index(drop=True)
    request_frame = pd.DataFrame(requests).sort_values("initialization_utc").reset_index(drop=True)
    if request_frame["initialization_utc"].duplicated().any():
        raise ValueError("CFSv2 plan unexpectedly maps multiple issues to one cycle")
    return join_frame, request_frame


def download_requests(
    requests: pd.DataFrame, raw_dir: Path, workers: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def download_one(row: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        planned_initialization = pd.Timestamp(row.initialization_utc)
        output = raw_dir / (
            f"cfsv2_{planned_initialization:%Y%m%d%H}_m{MEMBER}_uv10.grib2"
        )
        sidecar = output.with_suffix(".source.json")
        if output.is_file() and sidecar.is_file():
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            raw_file = cached["raw_file"]
            if hashlib.sha256(output.read_bytes()).hexdigest() == raw_file["sha256"]:
                actual = pd.Timestamp(raw_file["initialization_utc"])
                shift = int(
                    round(
                        (planned_initialization - actual).total_seconds() / 3600.0
                    )
                )
                raw_file.setdefault(
                    "planned_initialization_utc",
                    planned_initialization.isoformat(),
                )
                raw_file.setdefault("lead_shift_hours", shift)
                return cached["audit_row"], raw_file

        planned_leads = tuple(int(value) for value in str(row.lead_hours).split(","))
        source = None
        last_error: Exception | None = None
        # NCEI has isolated missing operational cycles. Only move backward in
        # time; the valid-time lead increases by the same amount.
        for lead_shift in range(0, 145, 6):
            initialization = planned_initialization - pd.Timedelta(hours=lead_shift)
            grib_url = object_url(initialization)
            inventory_url = object_url(initialization, "inv")
            leads = tuple(value + lead_shift for value in planned_leads)
            try:
                object_size, archive_modified = _head_archive(grib_url)
                inventory = _read_text(inventory_url)
                ranges = parse_inventory_ranges(inventory, object_size, leads)
                source = (
                    initialization,
                    lead_shift,
                    grib_url,
                    inventory_url,
                    archive_modified,
                    inventory,
                    ranges,
                )
                break
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                last_error = error
        if source is None:
            raise RuntimeError(
                f"CFSv2 source unavailable for planned cycle "
                f"{planned_initialization.isoformat()} after 144-hour backward search"
            ) from last_error
        (
            initialization,
            lead_shift,
            grib_url,
            inventory_url,
            archive_modified,
            inventory,
            ranges,
        ) = source
        payload = b"".join(
            _read_range(grib_url, item["offset"], item["end"])
            for item in ranges
        )
        temporary = output.with_suffix(".tmp")
        temporary.write_bytes(payload)
        _replace_with_retry(temporary, output)
        retrieved = datetime.now(timezone.utc).isoformat()
        digest = hashlib.sha256(payload).hexdigest()
        audit_row = {
            "prediction_reference_kst": pd.Timestamp(row.prediction_reference_kst).isoformat(),
            "initialization_utc": initialization.isoformat(),
            "public_availability_utc": (
                initialization + pd.Timedelta(DELAY)
            ).isoformat(),
        }
        raw_file = {
            "path": output.as_posix(),
            "source_url": grib_url,
            "inventory_url": inventory_url,
            "inventory_sha256": hashlib.sha256(inventory.encode("utf-8")).hexdigest(),
            "archive_last_modified_utc": archive_modified.isoformat(),
            "retrieved_at_utc": retrieved,
            "initialization_utc": initialization.isoformat(),
            "planned_initialization_utc": planned_initialization.isoformat(),
            "lead_shift_hours": int(lead_shift),
            "conservative_public_availability_utc": (
                initialization + pd.Timedelta(DELAY)
            ).isoformat(),
            "byte_ranges": ranges,
            "bytes": len(payload),
            "sha256": digest,
        }
        sidecar_tmp = sidecar.with_suffix(".tmp")
        sidecar_tmp.write_text(
            json.dumps(
                {"audit_row": audit_row, "raw_file": raw_file},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _replace_with_retry(sidecar_tmp, sidecar)
        return audit_row, raw_file

    audit_rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    rows = list(requests.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            audit, raw = future.result()
            audit_rows.append(audit)
            raw_files.append(raw)
            if completed % 50 == 0 or completed == len(rows):
                print(f"downloaded_or_verified={completed}/{len(rows)}", flush=True)
    raw_files.sort(key=lambda item: item["path"])
    return pd.DataFrame(audit_rows), raw_files


def apply_source_resolutions(
    joins: pd.DataFrame,
    requests: pd.DataFrame,
    raw_files: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_joins = joins.copy()
    resolved_requests = requests.copy()
    for item in raw_files:
        planned = pd.Timestamp(item["planned_initialization_utc"])
        actual = pd.Timestamp(item["initialization_utc"])
        shift = int(item["lead_shift_hours"])
        join_mask = pd.to_datetime(
            resolved_joins["initialization_utc"], utc=True
        ) == planned
        request_mask = pd.to_datetime(
            resolved_requests["initialization_utc"], utc=True
        ) == planned
        if int(join_mask.sum()) == 0 or int(request_mask.sum()) != 1:
            raise ValueError("CFSv2 source resolution does not match its plan")
        resolved_joins.loc[join_mask, "initialization_utc"] = actual
        resolved_joins.loc[join_mask, "public_availability_utc"] = (
            actual + pd.Timedelta(DELAY)
        )
        resolved_joins.loc[join_mask, "raw_lead_hours"] += shift
        resolved_joins.loc[join_mask, "lower_lead_hour"] += shift
        resolved_joins.loc[join_mask, "upper_lead_hour"] += shift
        original_leads = tuple(
            int(value)
            for value in str(
                resolved_requests.loc[request_mask, "lead_hours"].iloc[0]
            ).split(",")
        )
        resolved_requests.loc[request_mask, "planned_initialization_utc"] = planned
        resolved_requests.loc[request_mask, "initialization_utc"] = actual
        resolved_requests.loc[request_mask, "public_availability_utc"] = (
            actual + pd.Timedelta(DELAY)
        )
        resolved_requests.loc[request_mask, "lead_shift_hours"] = shift
        resolved_requests.loc[request_mask, "lead_hours"] = ",".join(
            str(value + shift) for value in original_leads
        )
        resolved_requests.loc[request_mask, "grib_url"] = item["source_url"]
        resolved_requests.loc[request_mask, "inventory_url"] = item[
            "inventory_url"
        ]
    return resolved_joins, resolved_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output-dir", default="artifacts_final/external_weather/noaa_cfsv2_2024"
    )
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--start-issue")
    parser.add_argument("--end-issue")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    metadata = pd.read_csv(
        args.forecast_metadata,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    joins, requests = build_plans(
        metadata, args.max_issues, args.start_issue, args.end_issue
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files: list[dict[str, Any]] = []
    causality: dict[str, Any] = {
        "rows": 0,
        "violations": 0,
        "minimum_availability_margin_minutes": -1,
    }
    if args.download:
        audit, raw_files = download_requests(requests, output_dir / "raw", args.workers)
        causality = audit_forecast_availability(audit)
        joins, requests = apply_source_resolutions(joins, requests, raw_files)
        root = Path.cwd().resolve()
        for item in raw_files:
            item["path"] = str(
                Path(item["path"]).resolve().relative_to(root)
            ).replace("\\", "/")
    joins.to_csv(output_dir / "join_plan.csv", index=False, encoding="utf-8-sig")
    requests.to_csv(output_dir / "request_plan.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "competition_eligible": bool(args.download and raw_files),
        "provider": "NOAA/NCEP/NCEI",
        "dataset": "CFSv2 operational 9-month forecast member 01 10 m wind",
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "US government public data; NOAA disclaimer applies",
        "license_url": LICENSE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "forecast_metadata": args.forecast_metadata,
            "issues": int(joins["data_available_kst_dtm"].nunique()),
            "targets": int(len(joins)),
            "objects": int(len(requests)),
            "member": MEMBER,
            "variables": ["UGRD:10 m above ground", "VGRD:10 m above ground"],
        },
        "availability_evidence": {
            "method": "provider_schedule_with_conservative_lag",
            "conservative_delay_minutes": int(DELAY.total_seconds() // 60),
            "evidence_url": PRODUCT_URL,
            "archive_evidence_url": DOCUMENTATION_URL,
            "explanation": (
                "NCEP documents four operational cycles daily and NCEI documents "
                "real-time NCEP distribution plus the historical operational archive; "
                "a deliberately conservative 30-hour bound is used instead of the "
                "later NCEI archival object timestamp."
            ),
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
