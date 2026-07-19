from __future__ import annotations

import pandas as pd
import pytest

from experiments.fetch_noaa_cfsv2_operational import (
    apply_source_resolutions,
    build_plans,
    object_url,
    parse_inventory_ranges,
)


def test_object_url_points_to_original_operational_cycle() -> None:
    url = object_url(pd.Timestamp("2023-12-29 18:00:00", tz="UTC"))
    assert "/2023/202312/20231229/2023122918/" in url
    assert url.endswith("wnd10m.01.2023122918.daily.grb2")


def test_plan_uses_thirty_hour_bound_and_six_hour_brackets() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": [
                "2024-01-01 01:00:00",
                "2024-01-01 02:00:00",
            ],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"] * 2,
        }
    )
    joins, requests = build_plans(frame)
    assert joins["initialization_utc"].nunique() == 1
    assert joins.loc[0, "initialization_utc"] == pd.Timestamp(
        "2023-12-29 18:00:00", tz="UTC"
    )
    assert joins[["lower_lead_hour", "upper_lead_hour"]].to_numpy().tolist() == [
        [42, 48],
        [42, 48],
    ]
    assert requests.loc[0, "lead_hours"] == "42,48"
    margin = (
        pd.Timestamp("2023-12-31 13:00:00", tz="Asia/Seoul").tz_convert("UTC")
        - joins.loc[0, "public_availability_utc"]
    )
    assert margin == pd.Timedelta(hours=4)


def test_inventory_parser_selects_exact_uv_leads() -> None:
    text = "\n".join(
        [
            "1:0:d=2023122918:UGRD:10 m above ground:42 hour fcst:",
            "2:100:d=2023122918:VGRD:10 m above ground:42 hour fcst:",
            "3:230:d=2023122918:UGRD:10 m above ground:48 hour fcst:",
            "4:350:d=2023122918:VGRD:10 m above ground:48 hour fcst:",
            "5:500:d=2023122918:TMP:2 m above ground:48 hour fcst:",
        ]
    )
    selected = parse_inventory_ranges(text, 700, (42, 48))
    assert [(item["offset"], item["end"]) for item in selected] == [
        (0, 99),
        (100, 229),
        (230, 349),
        (350, 499),
    ]


def test_inventory_parser_fails_closed_when_component_missing() -> None:
    text = "1:0:d=2023122918:UGRD:10 m above ground:42 hour fcst:\n"
    with pytest.raises(ValueError, match="missing requested"):
        parse_inventory_ranges(text, 100, (42,))


def test_backward_source_resolution_shifts_leads_without_changing_valid_time() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00"],
            "data_available_kst_dtm": ["2023-12-31 13:00:00"],
        }
    )
    joins, requests = build_plans(frame)
    planned = pd.Timestamp(requests.loc[0, "initialization_utc"])
    actual = planned - pd.Timedelta(hours=24)
    raw = [
        {
            "planned_initialization_utc": planned.isoformat(),
            "initialization_utc": actual.isoformat(),
            "lead_shift_hours": 24,
            "source_url": object_url(actual),
            "inventory_url": object_url(actual, "inv"),
        }
    ]
    resolved_joins, resolved_requests = apply_source_resolutions(
        joins, requests, raw
    )
    assert pd.Timestamp(resolved_joins.loc[0, "initialization_utc"]) == actual
    assert resolved_joins.loc[0, "lower_lead_hour"] == 66
    assert resolved_joins.loc[0, "upper_lead_hour"] == 72
    assert resolved_requests.loc[0, "lead_hours"] == "66,72"
    original_valid = planned + pd.Timedelta(hours=joins.loc[0, "raw_lead_hours"])
    resolved_valid = actual + pd.Timedelta(
        hours=resolved_joins.loc[0, "raw_lead_hours"]
    )
    assert original_valid == resolved_valid
