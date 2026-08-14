from __future__ import annotations

import pandas as pd

from experiments.fetch_noaa_gefs_spread import build_plans, parse_idx_ranges


def test_idx_parser_selects_uv10_ranges() -> None:
    index = "\n".join(
        [
            "66:900:d=2024010118:TMIN:2 m above ground:24 hour fcst:ens std dev",
            "67:1000:d=2024010118:UGRD:10 m above ground:24 hour fcst:ens std dev",
            "68:1250:d=2024010118:VGRD:10 m above ground:24 hour fcst:ens std dev",
            "69:1500:d=2024010118:APCP:surface:24 hour fcst:ens std dev",
        ]
    )
    selected = parse_idx_ranges(index, 2000)
    assert [(item["offset"], item["end"]) for item in selected] == [
        (1000, 1249),
        (1250, 1499),
    ]


def test_plan_uses_previous_18z_and_three_hour_brackets() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2025-01-01 01:00:00", "2025-01-01 02:00:00"],
            "data_available_kst_dtm": [
                "2024-12-31 13:00:00",
                "2024-12-31 13:00:00",
            ],
        }
    )
    joins, requests = build_plans(frame)
    assert joins["initialization_utc"].nunique() == 1
    assert joins.loc[0, "initialization_utc"] == pd.Timestamp(
        "2024-12-30 18:00", tz="UTC"
    )
    assert joins[["lower_lead_hour", "upper_lead_hour"]].to_numpy().tolist() == [
        [21, 24],
        [21, 24],
    ]
    assert requests["lead_hour"].tolist() == [21, 24]


def test_issue_date_filter_is_applied_before_planning() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00", "2025-01-01 01:00:00"],
            "data_available_kst_dtm": [
                "2023-12-31 13:00:00",
                "2024-12-31 13:00:00",
            ],
        }
    )
    joins, _ = build_plans(
        frame, start_issue="2024-01-01", end_issue="2024-12-31 23:59:59"
    )
    assert joins["forecast_kst_dtm"].tolist() == [pd.Timestamp("2025-01-01 01:00")]


def test_mean_product_can_request_single_screening_lead() -> None:
    frame = pd.DataFrame(
        {
            "forecast_kst_dtm": [f"2025-01-01 {hour:02d}:00:00" for hour in range(1, 24)],
            "data_available_kst_dtm": ["2024-12-31 13:00:00"] * 23,
        }
    )
    _, requests = build_plans(
        frame, product="geavg", requested_leads=(33,)
    )
    assert requests["lead_hour"].tolist() == [33]
    assert "/geavg.t18z.pgrb2a.0p50.f033" in requests.loc[0, "grib_url"]
