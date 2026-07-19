from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from agent_service.compliance import (
    audit_forecast_availability,
    audit_observation_availability,
    latest_safe_forecast_run,
    validate_external_data_manifest,
)


def test_baram_13_kst_cutoff_selects_previous_18z_with_conservative_delay() -> None:
    selected = latest_safe_forecast_run(
        "2025-01-12 13:00:00",
        conservative_publication_delay=timedelta(hours=6, minutes=10),
    )
    assert selected.initialization_utc == pd.Timestamp("2025-01-11 18:00", tz="UTC")
    assert selected.conservative_publication_utc == pd.Timestamp(
        "2025-01-12 00:10", tz="UTC"
    )


def test_availability_audit_allows_future_valid_time_but_not_future_publication() -> None:
    safe = pd.DataFrame(
        {
            "prediction_reference_kst": ["2025-01-12 13:00:00"],
            "initialization_utc": ["2025-01-11 18:00:00+00:00"],
            "public_availability_utc": ["2025-01-12 00:10:00+00:00"],
            "valid_time_utc": ["2025-01-13 01:00:00+00:00"],
        }
    )
    report = audit_forecast_availability(safe)
    assert report["violations"] == 0
    assert report["minimum_availability_margin_minutes"] == pytest.approx(230.0)

    unsafe = safe.copy()
    unsafe["public_availability_utc"] = "2025-01-12 04:01:00+00:00"
    with pytest.raises(ValueError, match="post-reference publication"):
        audit_forecast_availability(unsafe)


def test_observation_audit_applies_conservative_publication_lag() -> None:
    safe = pd.DataFrame(
        {
            "prediction_reference_kst": ["2024-01-01 13:00:00"],
            "observation_kst": ["2024-01-01 11:00:00"],
        }
    )
    report = audit_observation_availability(
        safe, conservative_publication_delay=timedelta(hours=2)
    )
    assert report["violations"] == 0
    assert report["minimum_availability_margin_minutes"] == 0.0

    unsafe = safe.copy()
    unsafe["observation_kst"] = "2024-01-01 11:01:00"
    with pytest.raises(ValueError, match="post-reference publication"):
        audit_observation_availability(
            unsafe, conservative_publication_delay=timedelta(hours=2)
        )


def test_external_manifest_requires_causality_and_verifies_checksum(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.grib2"
    raw_file.write_bytes(b"operational forecast bytes")
    digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "NOAA/NCEP",
        "dataset": "GEFS operational forecast",
        "source_type": "operational_forecast_archive",
        "documentation_url": "https://www.ncei.noaa.gov/",
        "license": "US government public data",
        "license_url": "https://www.noaa.gov/disclaimer",
        "retrieved_at_utc": "2026-07-18T03:00:00+00:00",
        "availability_evidence": {
            "method": "provider_schedule_with_conservative_lag",
            "conservative_delay_minutes": 370,
        },
        "causality_audit": {
            "rows": 24,
            "violations": 0,
            "minimum_availability_margin_minutes": 230.0,
        },
        "raw_files": [
            {
                "path": "raw.grib2",
                "source_url": "https://example.test/raw.grib2",
                "retrieved_at_utc": "2026-07-18T03:00:00+00:00",
                "sha256": digest,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_external_data_manifest(path, tmp_path)
    assert report["checked_file_count"] == 1

    manifest["causality_audit"]["violations"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="zero violations"):
        validate_external_data_manifest(path, tmp_path)
