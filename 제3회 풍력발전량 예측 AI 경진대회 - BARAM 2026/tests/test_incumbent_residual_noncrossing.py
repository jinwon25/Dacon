from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.incumbent_residual_noncrossing import (
    DIRECT_ACTION_RATIOS,
    _metric,
    apply_policy,
    direct_utility_advantage,
    interval_month,
    residual_bayes_action,
)
from src.metrics import CAPACITY_KWH, evaluate_competition


def test_interval_month_assigns_midnight_to_previous_interval() -> None:
    index = pd.DatetimeIndex(
        ["2024-02-01 00:00:00", "2025-01-01 00:00:00"]
    )
    assert interval_month(index).tolist() == [1, 12]


def test_residual_action_is_bounded_around_incumbent() -> None:
    quantiles = np.tile(np.arange(0.05, 1.0, 0.10), (12, 1))
    incumbent = np.full(12, 10_000.0)
    action, advantage, diagnostics = residual_bayes_action(
        quantiles,
        incumbent,
        capacity=21_600.0,
        mean_eligible_generation=10_000.0,
    )
    assert action.shape == advantage.shape == incumbent.shape
    assert np.all(np.abs(action - incumbent) <= 0.0400001 * 21_600.0)
    assert diagnostics["maximum_action_ratio"] <= 0.0400001


def test_policy_never_moves_nonpositive_advantage_rows() -> None:
    incumbent = {
        "kpx_group_1": np.asarray([1_000.0, 1_000.0, 1_000.0]),
        "kpx_group_2": np.asarray([2_000.0, 2_000.0, 2_000.0]),
        "kpx_group_3": np.asarray([3_000.0, 3_000.0, 3_000.0]),
    }
    action = {
        target: values + 400.0 for target, values in incumbent.items()
    }
    advantage = {
        target: np.asarray([-0.1, 0.0, 0.2]) for target in incumbent
    }
    candidate, gates = apply_policy(
        incumbent,
        action,
        advantage,
        coverage=0.40,
        weight=0.50,
    )
    for target in incumbent:
        assert gates[target].tolist() == [False, False, True]
        assert np.allclose(
            candidate[target], incumbent[target] + [0.0, 0.0, 200.0]
        )


def test_metric_matches_official_when_every_group_is_valid() -> None:
    truth = {
        target: np.array([0.20, 0.30]) * capacity
        for target, capacity in CAPACITY_KWH.items()
    }
    prediction = {
        target: values * 0.97
        for target, values in truth.items()
    }
    rows = np.array([True, True])

    observed = _metric(truth, prediction, rows)
    expected = evaluate_competition(truth, prediction)

    for component in ("score", "one_minus_nmae", "ficr"):
        assert observed[component] == expected[component]


def test_metric_skips_group_without_eligible_rows_in_short_slice() -> None:
    truth = {
        target: np.array([0.20, 0.30]) * capacity
        for target, capacity in CAPACITY_KWH.items()
    }
    truth["kpx_group_3"] = (
        np.array([0.01, 0.02]) * CAPACITY_KWH["kpx_group_3"]
    )
    prediction = {
        target: values * 0.97
        for target, values in truth.items()
    }

    observed = _metric(truth, prediction, np.array([True, True]))

    assert set(observed["groups"]) == {"kpx_group_1", "kpx_group_2"}
    assert np.isfinite(observed["score"])


def test_direct_utility_advantage_uses_incumbent_as_exact_zero() -> None:
    advantage = direct_utility_advantage(
        np.array([10_000.0]),
        np.array([8_000.0]),
        capacity=20_000.0,
        mean_eligible_generation=10_000.0,
    )
    zero = int(np.argmin(np.abs(DIRECT_ACTION_RATIOS)))
    positive_four_percent = int(
        np.argmin(np.abs(DIRECT_ACTION_RATIOS - 0.04))
    )

    assert advantage[0, zero] == 0.0
    assert advantage[0, positive_four_percent] > 0.0
