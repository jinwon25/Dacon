from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.fetch_open_meteo_jma_msm import (
    build_causal_context,
    build_url,
    parse_locations,
)


def _payload(times: pd.DatetimeIndex) -> bytes:
    values = np.arange(len(times), dtype=float) + 1.0
    location = {
        "latitude": 37.3,
        "longitude": 128.9375,
        "elevation": 1114.0,
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "wind_speed_10m_previous_day1": values.tolist(),
            "wind_direction_10m_previous_day1": [270.0] * len(times),
            "wind_speed_10m_previous_day2": (values / 2.0).tolist(),
            "wind_direction_10m_previous_day2": [180.0] * len(times),
        },
    }
    return json.dumps(location).encode()


def _thermodynamic_payload(times: pd.DatetimeIndex) -> bytes:
    raw = json.loads(_payload(times))
    size = len(times)
    raw["hourly"].update(
        {
            "temperature_2m_previous_day1": [-5.0] * size,
            "temperature_2m_previous_day2": [-6.0] * size,
            "surface_pressure_previous_day1": [920.0] * size,
            "surface_pressure_previous_day2": [919.0] * size,
            "relative_humidity_2m_previous_day1": [50.0] * size,
            "relative_humidity_2m_previous_day2": [55.0] * size,
        }
    )
    return json.dumps(raw).encode()


def _day3_payload(times: pd.DatetimeIndex) -> bytes:
    raw = json.loads(_payload(times))
    size = len(times)
    raw["hourly"].update(
        {
            "wind_speed_10m_previous_day3": [0.75] * size,
            "wind_direction_10m_previous_day3": [225.0] * size,
        }
    )
    return json.dumps(raw).encode()


def test_jma_previous_run_policy_is_causal_across_baram_leads() -> None:
    times = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 14:00")]
    )
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": [
                times[0] - pd.Timedelta(hours=12),
                times[1] - pd.Timedelta(hours=25),
            ],
        }
    )
    locations = parse_locations(_payload(times))

    context, audit = build_causal_context(locations, primary)

    assert audit["violations"] == 0
    assert audit["day1_rows"] == 1
    assert audit["day2_rows"] == 1
    assert audit["minimum_availability_margin_minutes"] == 570.0
    assert context["kma_um_ctx_jma_msm__selected_offset_hours"].tolist() == [
        24.0,
        48.0,
    ]
    assert context.loc[0, "kma_um_ctx_jma_msm_g1__safe_speed10"] == 1.0
    assert context.loc[1, "kma_um_ctx_jma_msm_g1__safe_speed10"] == 1.0


def test_build_url_pins_model_offsets_and_nearest_cell() -> None:
    url = build_url((37.283,), (128.95,), "2024-01-01", "2024-12-31")
    assert "models=jma_msm" in url
    assert "previous_day1" in url
    assert "previous_day2" in url
    assert "cell_selection=nearest" in url


def test_gsm_url_and_conservative_delay_are_model_specific() -> None:
    url = build_url(
        (37.5,),
        (129.0,),
        "2024-01-01",
        "2024-12-31",
        model="jma_gsm",
    )
    assert "models=jma_gsm" in url

    times = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 08:00")]
    )
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": [
                times[0] - pd.Timedelta(hours=18),
                times[1] - pd.Timedelta(hours=25),
            ],
        }
    )
    locations = parse_locations(_payload(times))
    context, audit = build_causal_context(
        locations,
        primary,
        model="jma_gsm",
    )

    assert audit["publication_delay_minutes"] == 360
    assert audit["day1_maximum_safe_lead_hours"] == 18
    assert context["kma_um_ctx_jma_gsm__selected_offset_hours"].tolist() == [
        24.0,
        48.0,
    ]


def test_compact_stencil_emits_fixed_spatial_features() -> None:
    times = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 02:00")]
    )
    locations = []
    for position in range(9):
        parsed = parse_locations(_payload(times))[0]
        for column in parsed.columns:
            if column != "time":
                parsed[column] = parsed[column] + position / 10.0
        locations.append(parsed)
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": times - pd.Timedelta(hours=12),
        }
    )

    context, audit = build_causal_context(
        locations,
        primary,
        compact_stencil=True,
    )

    assert audit["feature_strategy"] == "compact_3x3_stencil"
    assert len(context.columns) - 2 == 41
    assert (
        "kma_um_ctx_jma_msm_stencil__safe_speed10__east_west"
        in context.columns
    )


def test_thermodynamic_stencil_adds_causal_density_features() -> None:
    times = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 02:00")]
    )
    locations = [
        parse_locations(
            _thermodynamic_payload(times),
            include_thermodynamics=True,
        )[0]
        for _ in range(9)
    ]
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": times - pd.Timedelta(hours=12),
        }
    )

    context, audit = build_causal_context(
        locations,
        primary,
        compact_stencil=True,
        include_thermodynamics=True,
    )

    assert audit["thermodynamic_features"] is True
    assert (
        "kma_um_ctx_jma_msm_stencil__safe_air_density__centre"
        in context
    )
    assert (
        "kma_um_ctx_jma_msm_stencil__safe_density_speed10__mean"
        in context
    )
    assert np.isfinite(context.select_dtypes("number").to_numpy()).all()


def test_day3_stencil_adds_safe_revision_features() -> None:
    times = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 01:00"), pd.Timestamp("2024-01-02 02:00")]
    )
    locations = [
        parse_locations(
            _day3_payload(times),
            include_previous_day3=True,
        )[0]
        for _ in range(9)
    ]
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": times,
            "data_available_kst_dtm": times - pd.Timedelta(hours=12),
        }
    )

    context, audit = build_causal_context(
        locations,
        primary,
        compact_stencil=True,
        model="jma_gsm",
        include_previous_day3=True,
    )

    assert audit["previous_day3_wind_features"] is True
    assert "kma_um_ctx_jma_gsm_stencil__day3_speed10__mean" in context
    assert (
        "kma_um_ctx_jma_gsm_stencil__day2_minus_day3_u__centre"
        in context
    )
