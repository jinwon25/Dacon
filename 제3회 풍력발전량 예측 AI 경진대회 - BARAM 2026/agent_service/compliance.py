from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from agent_service.contracts import resolve_inside


UTC = ZoneInfo("UTC")
KST = ZoneInfo("Asia/Seoul")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SOURCE_TYPES = {
    "operational_forecast_archive",
    "timestamped_operational_catalog",
    "timestamped_observation_archive",
}
ALLOWED_AVAILABILITY_EVIDENCE = {
    "provider_timestamped_catalog",
    "archive_object_timestamp_with_conservative_lag",
    "provider_schedule_with_conservative_lag",
    "observation_timestamp_with_conservative_lag",
}


@dataclass(frozen=True)
class SafeForecastRun:
    initialization_utc: pd.Timestamp
    conservative_publication_utc: pd.Timestamp
    prediction_reference_kst: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "initialization_utc": self.initialization_utc.isoformat(),
            "conservative_publication_utc": (
                self.conservative_publication_utc.isoformat()
            ),
            "prediction_reference_kst": self.prediction_reference_kst.isoformat(),
        }


def _timestamp(value: object, default_timezone: ZoneInfo) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("timestamp must not be missing")
    if result.tzinfo is None:
        result = result.tz_localize(default_timezone)
    return result


def latest_safe_forecast_run(
    prediction_reference_kst: object,
    *,
    cycle_hours_utc: Iterable[int] = (0, 6, 12, 18),
    conservative_publication_delay: timedelta = timedelta(hours=6, minutes=10),
    lookback_days: int = 3,
) -> SafeForecastRun:
    """Select the newest model cycle conservatively public before prediction time.

    The delay is deliberately a hard availability bound, not an estimate of when
    a forecast *usually* appears. A provider timestamp may replace it only when
    that timestamp is retained in the provenance manifest.
    """

    reference_kst = _timestamp(prediction_reference_kst, KST).tz_convert(KST)
    reference_utc = reference_kst.tz_convert(UTC)
    hours = sorted({int(hour) for hour in cycle_hours_utc})
    if not hours or any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError("cycle_hours_utc must contain hours between 0 and 23")
    if conservative_publication_delay < timedelta(0):
        raise ValueError("conservative_publication_delay must not be negative")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")

    day = reference_utc.normalize()
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for offset in range(lookback_days + 1):
        candidate_day = day - pd.Timedelta(days=offset)
        for hour in hours:
            initialization = candidate_day + pd.Timedelta(hours=hour)
            publication = initialization + conservative_publication_delay
            if publication <= reference_utc:
                candidates.append((initialization, publication))
    if not candidates:
        raise ValueError("no forecast cycle is safely public before the reference time")
    initialization, publication = max(candidates, key=lambda item: item[0])
    return SafeForecastRun(initialization, publication, reference_kst)


def audit_forecast_availability(
    frame: pd.DataFrame,
    *,
    reference_column: str = "prediction_reference_kst",
    availability_column: str = "public_availability_utc",
    initialization_column: str = "initialization_utc",
) -> dict[str, Any]:
    """Fail closed when any external forecast was unavailable at prediction time."""

    required = {reference_column, availability_column, initialization_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"availability audit is missing columns: {missing}")
    if frame.empty:
        raise ValueError("availability audit requires at least one row")

    reference = pd.to_datetime(frame[reference_column], errors="coerce")
    if reference.dt.tz is None:
        reference = reference.dt.tz_localize(KST)
    else:
        reference = reference.dt.tz_convert(KST)
    availability = pd.to_datetime(
        frame[availability_column], errors="coerce", utc=True
    )
    initialization = pd.to_datetime(
        frame[initialization_column], errors="coerce", utc=True
    )
    if reference.isna().any() or availability.isna().any() or initialization.isna().any():
        raise ValueError("availability audit timestamps must all be parseable")

    reference_utc = reference.dt.tz_convert(UTC)
    future = availability > reference_utc
    creation_after_publication = initialization > availability
    if future.any():
        raise ValueError(
            f"external forecast has {int(future.sum())} post-reference publication rows"
        )
    if creation_after_publication.any():
        raise ValueError(
            "forecast initialization cannot be later than public availability"
        )
    margins = (reference_utc - availability).dt.total_seconds() / 60.0
    return {
        "rows": int(len(frame)),
        "violations": 0,
        "reference_column": reference_column,
        "availability_column": availability_column,
        "initialization_column": initialization_column,
        "minimum_availability_margin_minutes": float(margins.min()),
        "maximum_availability_margin_minutes": float(margins.max()),
    }


def audit_observation_availability(
    frame: pd.DataFrame,
    *,
    conservative_publication_delay: timedelta,
    reference_column: str = "prediction_reference_kst",
    observation_column: str = "observation_kst",
) -> dict[str, Any]:
    """Fail closed unless every observation was public before prediction time.

    Observation archives usually expose the measurement timestamp rather than a
    historical object-creation timestamp.  A fixed, documented publication lag
    is therefore added before comparing it with the competition reference time.
    This also makes retrospective feature generation use exactly the same
    information boundary as deployment.
    """

    if conservative_publication_delay < timedelta(0):
        raise ValueError("conservative publication delay must not be negative")
    required = {reference_column, observation_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"observation availability audit is missing columns: {missing}")
    if frame.empty:
        raise ValueError("observation availability audit requires at least one row")

    reference = pd.to_datetime(frame[reference_column], errors="coerce")
    if reference.dt.tz is None:
        reference = reference.dt.tz_localize(KST)
    else:
        reference = reference.dt.tz_convert(KST)
    observation = pd.to_datetime(frame[observation_column], errors="coerce")
    if observation.dt.tz is None:
        observation = observation.dt.tz_localize(KST)
    else:
        observation = observation.dt.tz_convert(KST)
    if reference.isna().any() or observation.isna().any():
        raise ValueError("observation audit timestamps must all be parseable")

    public_at = observation + conservative_publication_delay
    future = public_at > reference
    if future.any():
        raise ValueError(
            f"external observation has {int(future.sum())} post-reference publication rows"
        )
    margins = (reference - public_at).dt.total_seconds() / 60.0
    return {
        "rows": int(len(frame)),
        "violations": 0,
        "reference_column": reference_column,
        "observation_column": observation_column,
        "conservative_delay_minutes": int(
            conservative_publication_delay.total_seconds() // 60
        ),
        "minimum_availability_margin_minutes": float(margins.min()),
        "maximum_availability_margin_minutes": float(margins.max()),
    }


def validate_external_data_manifest(
    manifest_path: Path,
    project_root: Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate provenance required before an external-data run can be promoted."""

    path = resolve_inside(project_root, str(manifest_path), "external_manifest_path")
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def required_text(name: str) -> str:
        value = str(raw.get(name, "")).strip()
        if not value:
            errors.append(f"{name} is required")
        return value

    required_text("provider")
    required_text("dataset")
    required_text("documentation_url")
    required_text("license")
    required_text("license_url")
    required_text("retrieved_at_utc")
    if raw.get("competition_eligible") is not True:
        errors.append("competition_eligible must be true")
    if raw.get("source_type") not in ALLOWED_SOURCE_TYPES:
        errors.append("source_type is not an allowed operational source")

    availability = raw.get("availability_evidence", {})
    if not isinstance(availability, dict):
        errors.append("availability_evidence must be an object")
        availability = {}
    if availability.get("method") not in ALLOWED_AVAILABILITY_EVIDENCE:
        errors.append("availability evidence is not strong enough")
    try:
        delay = int(availability.get("conservative_delay_minutes", -1))
        if delay < 0:
            errors.append("conservative_delay_minutes must be non-negative")
    except (TypeError, ValueError):
        errors.append("conservative_delay_minutes must be an integer")

    causality = raw.get("causality_audit", {})
    if not isinstance(causality, dict):
        errors.append("causality_audit must be an object")
        causality = {}
    if int(causality.get("rows", 0) or 0) <= 0:
        errors.append("causality_audit.rows must be positive")
    if int(causality.get("violations", -1) or 0) != 0:
        errors.append("causality_audit must have zero violations")
    try:
        margin = float(causality.get("minimum_availability_margin_minutes", -1))
        if margin < 0:
            errors.append("minimum availability margin must be non-negative")
    except (TypeError, ValueError):
        errors.append("minimum availability margin must be numeric")

    raw_files = raw.get("raw_files", [])
    if not isinstance(raw_files, list) or not raw_files:
        errors.append("raw_files must contain at least one source file")
        raw_files = []
    checked_files = 0
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            errors.append(f"raw_files[{index}] must be an object")
            continue
        digest = str(item.get("sha256", "")).lower()
        if not SHA256.fullmatch(digest):
            errors.append(f"raw_files[{index}].sha256 is invalid")
        for name in ("path", "source_url", "retrieved_at_utc"):
            if not str(item.get(name, "")).strip():
                errors.append(f"raw_files[{index}].{name} is required")
        if verify_files and str(item.get("path", "")).strip():
            local = resolve_inside(project_root, item["path"], f"raw_files[{index}].path")
            if not local.is_file():
                errors.append(f"raw_files[{index}] does not exist")
            elif SHA256.fullmatch(digest):
                actual = hashlib.sha256(local.read_bytes()).hexdigest()
                if actual != digest:
                    errors.append(f"raw_files[{index}] checksum mismatch")
                else:
                    checked_files += 1

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "manifest": str(path.relative_to(project_root)),
        "provider": raw["provider"],
        "dataset": raw["dataset"],
        "raw_file_count": len(raw_files),
        "checked_file_count": checked_files,
        "minimum_availability_margin_minutes": margin,
    }
