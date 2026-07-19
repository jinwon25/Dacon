from __future__ import annotations

import pandas as pd
import pytest

from experiments.fetch_kma_um_global import (
    RequestSpec,
    build_features,
    build_url,
    parse_response,
    request_specs,
)


def test_request_specs_use_previous_day_12z_with_twelve_hour_delay() -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.date_range(
                "2024-01-01 01:00:00", periods=24, freq="h"
            ),
            "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00:00"),
        }
    )
    specs, audit = request_specs(metadata, ((37.28, 128.96),), 12.0)
    assert {spec.initialization_utc.hour for spec in specs} == {12}
    assert {spec.initialization_utc.day for spec in specs} == {30}
    assert min(spec.lead_hour for spec in specs) == 27
    assert max(spec.lead_hour for spec in specs) == 51
    assert len(audit) == 1
    margin = (
        pd.Timestamp("2023-12-31 13:00:00", tz="Asia/Seoul").tz_convert("UTC")
        - specs[0].public_availability_utc
    )
    assert margin == pd.Timedelta(hours=4)


def test_url_redacts_api_key() -> None:
    spec = RequestSpec(
        pd.Timestamp("2023-12-31 13:00:00"),
        pd.Timestamp("2023-12-30 12:00:00", tz="UTC"),
        pd.Timestamp("2023-12-31 00:00:00", tz="UTC"),
        54,
        37.28,
        128.96,
        1,
    )
    url = build_url(spec, "private-key")
    redacted = build_url(spec, "private-key", redact=True)
    assert "private-key" in url
    assert "private-key" not in redacted
    assert "%3Credacted%3E" in redacted
    assert "varn=2002%2C2003" in redacted


def test_parse_response_supports_tagged_and_table_formats() -> None:
    tagged = "VARN=2002 LEVEL=10 VALUS=1.25\nVARN=2003 LEVEL=10 VALUS=-2.5"
    assert parse_response(tagged) == {2002: 1.25, 2003: -2.5}
    table = (
        "TMFC TMEF VARN LEVEL VALUS\n"
        "2023123012 2024010118 2002 10 1.5\n"
        "2023123012 2024010118 2003 10 -3.0\n"
    )
    assert parse_response(table) == {2002: 1.5, 2003: -3.0}


def test_parse_response_fails_closed_on_incomplete_data() -> None:
    with pytest.raises(ValueError, match="both 10 m wind components"):
        parse_response("2023123012 2024010118 2002 10 1.5")


def test_build_features_uses_only_cycle_public_by_reference(tmp_path) -> None:
    metadata = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.date_range(
                "2024-01-01 01:00:00", periods=3, freq="h"
            ),
            "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00:00"),
        }
    )
    records = []
    for initialization, availability, offset in (
        ("2023-12-30 12:00:00+00:00", "2023-12-31 00:00:00+00:00", 0.0),
        # This newer cycle contains deliberately extreme values but was not yet public.
        ("2023-12-31 00:00:00+00:00", "2023-12-31 12:00:00+00:00", 100.0),
    ):
        initialization_ts = pd.Timestamp(initialization)
        for valid, value in (
            ("2023-12-31 15:00:00+00:00", 1.0 + offset),
            ("2023-12-31 18:00:00+00:00", 4.0 + offset),
        ):
            path = tmp_path / f"raw_{len(records)}.txt"
            path.write_text(
                f"VARN=2002 VALUS={value}\nVARN=2003 VALUS={value * 2}",
                encoding="utf-8",
            )
            records.append(
                {
                    "path": path.as_posix(),
                    "initialization_utc": initialization_ts.isoformat(),
                    "valid_utc": pd.Timestamp(valid).isoformat(),
                    "point_id": 1,
                    "latitude": 37.28,
                    "longitude": 128.96,
                    "conservative_public_availability_utc": availability,
                }
            )

    output = tmp_path / "features.csv"
    result = build_features(metadata, records, output)

    assert output.is_file()
    assert len(result) == 3
    assert result["initialization_utc"].nunique() == 1
    assert result["initialization_utc"].iloc[0].startswith("2023-12-30T12:00:00")
    assert result["kma_um_u10"].tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert result["kma_um_v10"].tolist() == pytest.approx([4.0, 6.0, 8.0])
