from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.fetch_kma_asos_observations import (
    RequestSpec,
    build_issue_features,
    build_url,
    parse_response,
    request_specs,
)


def _response(rows: list[str]) -> str:
    header = (
        "# YYMMDDHHMI STN WD WS GST_WD GST_WS PA PS PT PR "
        "TA TD HM RN RN_DAY RN_INT"
    )
    return "\n".join(["# START7777", header, *rows, "#7777END"])


def test_asos_request_specs_apply_lag_and_cover_history() -> None:
    issues = pd.DatetimeIndex(["2024-01-31 13:00", "2024-02-02 13:00"])
    specs = request_specs(
        issues,
        (216,),
        publication_delay_minutes=120,
        history_hours=24,
        chunk_days=1,
    )
    assert specs[0].start_kst == pd.Timestamp(
        "2024-01-30 11:00", tz="Asia/Seoul"
    )
    assert specs[-1].end_kst == pd.Timestamp(
        "2024-02-02 11:00", tz="Asia/Seoul"
    )
    assert all(spec.station_id == 216 for spec in specs)


def test_asos_url_never_exposes_key_in_redacted_form() -> None:
    spec = RequestSpec(
        216,
        pd.Timestamp("2024-01-01 00:00", tz="Asia/Seoul"),
        pd.Timestamp("2024-01-30 23:00", tz="Asia/Seoul"),
    )
    assert "private-key" in build_url(spec, "private-key")
    redacted = build_url(spec, "private-key", redact=True)
    assert "private-key" not in redacted
    assert "%3Credacted%3E" in redacted
    assert "stn=216" in redacted


def test_parse_asos_response_and_reject_wrong_station() -> None:
    text = _response(
        ["202401010900 216 270 4.0 280 8.0 930 1010 1 1.2 -5 -8 60 0 0 0"]
    )
    frame = parse_response(text, expected_station=216)
    assert frame.loc[0, "TM"] == pd.Timestamp("2024-01-01 09:00")
    assert frame.loc[0, "WS"] == 4.0
    assert frame.loc[0, "TA"] == -5.0
    with pytest.raises(ValueError, match="unexpected station"):
        parse_response(text, expected_station=217)


def test_parse_official_grouped_fixed_width_header() -> None:
    header = (
        "# YYMMDDHHMI STN WD WS GST GST GST PA PS PT PR TA TD HM PV "
        "RN RN RN RN SD SD SD WC WP WW CA CA CH CT CT CT CT VS SS SI "
        "ST TS TE TE TE TE ST WH BF IR IX"
    )
    row = " ".join(
        [
            "202401010900",
            "216",
            "27",
            "4.0",
            "28",
            "8.0",
            "830",
            "930",
            "1010",
            "1",
            "1.2",
            "-5",
            "-8",
            "60",
            "4",
            "0",
            "0",
            "0",
            "0",
            "-9",
            "-9",
            "-9",
            "-9",
            "-9",
            "-9",
            "0",
            "0",
            "-9",
            "-9",
            "-9",
            "-9",
            "-9",
            "2000",
            "0",
            "0",
            "-9",
            "-1",
            "-9",
            "-9",
            "-9",
            "-9",
            "-9",
            "-9",
            "1",
            "3",
            "2",
        ]
    )
    frame = parse_response("\n".join([header, row, "#7777END"]), expected_station=216)
    assert frame.loc[0, "GST_WS"] == 8.0
    assert frame.loc[0, "RN_DAY"] == 0.0
    assert frame.loc[0, "WS"] == 4.0


def test_issue_features_exclude_observations_inside_safety_lag() -> None:
    timestamps = pd.date_range("2024-01-01 00:00", "2024-01-01 12:00", freq="h")
    observations = pd.DataFrame(
        {
            "TM": timestamps,
            "STN": 216,
            "WD": 270.0,
            "WS": np.where(timestamps.hour == 12, 100.0, timestamps.hour + 1.0),
            "GST_WD": 270.0,
            "GST_WS": np.where(timestamps.hour == 12, 120.0, timestamps.hour + 2.0),
            "PA": 930.0,
            "PS": 1010.0,
            "PT": 1.0,
            "PR": 0.1,
            "TA": 5.0,
            "TD": 1.0,
            "HM": 60.0,
            "RN": 0.0,
            "RN_DAY": 0.0,
            "RN_INT": 0.0,
        }
    )
    features, audit = build_issue_features(
        pd.DatetimeIndex(["2024-01-01 13:00"]),
        observations,
        publication_delay_minutes=120,
    )
    row = features.iloc[0]
    assert row["latest_observation_kst"] == pd.Timestamp("2024-01-01 11:00")
    assert row["asos_stn216__h03__count"] == 3.0
    assert row["asos_stn216__h03__ws_mean"] == pytest.approx(11.0)
    assert row["asos_stn216__h03__u_mean"] == pytest.approx(11.0)
    assert audit["observation_kst"].max() == pd.Timestamp("2024-01-01 11:00")
