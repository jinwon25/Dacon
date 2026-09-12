from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.fetch_kma_village_forecast import (
    CATEGORIES,
    VillageRequestSpec,
    build_features,
    build_url,
    latest_safe_village_cycle,
    parse_response,
    validate_response,
    village_request_specs,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.date_range(
                "2024-01-01 01:00:00", periods=3, freq="h"
            ),
            "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00:00"),
        }
    )


def _payload() -> str:
    items = []
    for forecast in pd.date_range("2024-01-01 01:00:00", periods=3, freq="h"):
        for offset, category in enumerate(CATEGORIES):
            items.append(
                {
                    "baseDate": "20231231",
                    "baseTime": "1100",
                    "category": category,
                    "fcstDate": forecast.strftime("%Y%m%d"),
                    "fcstTime": forecast.strftime("%H%M"),
                    "fcstValue": str(offset + forecast.hour / 10),
                    "nx": 94,
                    "ny": 121,
                }
            )
    return json.dumps(
        {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                "body": {
                    "dataType": "JSON",
                    "items": {"item": items},
                    "pageNo": 1,
                    "numOfRows": 2000,
                    "totalCount": len(items),
                },
            }
        }
    )


def test_latest_cycle_uses_11_kst_issue_at_13_kst_reference() -> None:
    base, public = latest_safe_village_cycle("2023-12-31 13:00:00")
    assert base == pd.Timestamp("2023-12-31 11:00", tz="Asia/Seoul")
    assert public == pd.Timestamp("2023-12-31 12:00", tz="Asia/Seoul")


def test_specs_are_causal_and_url_redacts_key() -> None:
    specs, audit = village_request_specs(_metadata())
    assert len(specs) == 1
    assert len(audit) == 1
    assert audit.loc[0, "public_availability_utc"] < audit.loc[
        0, "prediction_reference_kst"
    ].tz_convert("UTC")
    redacted = build_url(specs[0], "private-key", redact=True)
    assert "private-key" not in redacted
    assert "%3Credacted%3E" in redacted
    assert "base_time=1100" in redacted


def test_parser_and_validator_retain_complete_wind_vectors() -> None:
    parsed = parse_response(_payload())
    assert len(parsed) == 3 * len(CATEGORIES)
    spec = VillageRequestSpec(
        reference_kst=pd.Timestamp("2023-12-31 13:00", tz="Asia/Seoul"),
        base_kst=pd.Timestamp("2023-12-31 11:00", tz="Asia/Seoul"),
        public_availability_kst=pd.Timestamp(
            "2023-12-31 12:00", tz="Asia/Seoul"
        ),
        nx=94,
        ny=121,
    )
    validate_response(parsed, spec)


def test_parser_rejects_truncated_page() -> None:
    payload = json.loads(_payload())
    payload["response"]["body"]["totalCount"] += 1
    with pytest.raises(ValueError, match="paginated and is incomplete"):
        parse_response(json.dumps(payload))


def test_build_features_joins_targets_and_derives_vector_delta(tmp_path: Path) -> None:
    raw = tmp_path / "village.json"
    raw.write_text(_payload(), encoding="utf-8")
    record = {
        "path": raw.as_posix(),
        "prediction_reference_kst": pd.Timestamp(
            "2023-12-31 13:00", tz="Asia/Seoul"
        ).isoformat(),
        "base_kst": pd.Timestamp(
            "2023-12-31 11:00", tz="Asia/Seoul"
        ).isoformat(),
        "conservative_public_availability_utc": pd.Timestamp(
            "2023-12-31 03:00", tz="UTC"
        ).isoformat(),
    }
    result = build_features(_metadata(), [record], tmp_path / "features.csv")
    assert len(result) == 3
    assert result[list(CATEGORIES.values())].notna().all().all()
    assert result["kma_village_vector_speed10"].gt(0).all()
    assert result["kma_village_speed_vector_delta"].notna().all()
    assert (tmp_path / "features.csv").is_file()
