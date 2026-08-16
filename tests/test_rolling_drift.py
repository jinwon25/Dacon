import numpy as np
import pandas as pd

from src.calibration import apply_logit_offset, forecast_base_rate
from src.metrics import probability_logit
from src.rolling_drift import build_rolling_damped_ensemble


def test_fixed_damped_ensemble_is_probability_level_average():
    raw = np.array([0.2, 0.5, 0.8])
    rates = pd.Series(
        [0.56, 0.53, 0.52],
        index=[2019, 2020, 2021],
        dtype=float,
    )
    methods = ["damped_3_0.5", "damped_3_0.8"]
    actual, details = build_rolling_damped_ensemble(
        raw, rates, 2022, methods, [0.5, 0.5]
    )
    latest_logit = probability_logit(np.array([rates.iloc[-1]]))[0]
    members = []
    for method in methods:
        forecast = forecast_base_rate(rates, 2022, method)
        offset = probability_logit(np.array([forecast]))[0] - latest_logit
        members.append(apply_logit_offset(raw, float(offset)))
    np.testing.assert_allclose(actual, np.mean(np.column_stack(members), axis=1))
    assert [item["method"] for item in details] == methods
    assert [item["active_weight"] for item in details] == [0.5, 0.5]


def test_fixed_damped_ensemble_falls_back_without_three_seasons():
    raw = np.array([0.2, 0.5, 0.8])
    rates = pd.Series([0.56, 0.53], index=[2019, 2020], dtype=float)
    actual, details = build_rolling_damped_ensemble(
        raw,
        rates,
        2021,
        ["damped_3_0.5", "damped_3_0.8"],
        [0.5, 0.5],
    )
    np.testing.assert_array_equal(actual, raw)
    assert details == []


def test_fixed_damped_ensemble_rejects_nonhistorical_rates():
    raw = np.array([0.2, 0.5, 0.8])
    rates = pd.Series([0.56, 0.53, 0.52], index=[2020, 2021, 2022])
    try:
        build_rolling_damped_ensemble(
            raw,
            rates,
            2022,
            ["damped_3_0.5", "damped_3_0.8"],
            [0.5, 0.5],
        )
    except ValueError as error:
        assert "strictly before" in str(error)
    else:
        raise AssertionError("forecast-year rates must be rejected")
