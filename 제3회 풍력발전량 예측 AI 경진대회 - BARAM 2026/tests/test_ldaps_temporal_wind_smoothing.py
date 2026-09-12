from __future__ import annotations

import numpy as np

from experiments.ldaps_temporal_wind_smoothing import (
    CAPACITY,
    SmoothingPolicy,
    apply_smoothing_policy,
    smooth_within_issue,
)


def test_smoothing_does_not_cross_issue_boundaries() -> None:
    values = np.asarray([0.0, 3.0, 6.0, 60.0, 63.0, 66.0])
    issues = np.asarray(["a", "a", "a", "b", "b", "b"])
    result = smooth_within_issue(values, issues)
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0, 61.0, 63.0, 65.0])


def test_smoothing_policy_uses_difference_and_bounds_increment() -> None:
    raw = np.asarray([5_000.0, 5_000.0, 5_000.0])
    smoothed = np.asarray([8_000.0, 2_000.0, 8_000.0])
    incumbent = np.asarray([6_000.0, 6_000.0, 6_000.0])
    policy = SmoothingPolicy(
        direction="up",
        coverage=0.10,
        minimum_base_ratio=0.20,
        maximum_base_ratio=0.80,
        minimum_smoothing_difference_kwh=1_000.0,
        alpha=1.0,
    )
    candidate, gate = apply_smoothing_policy(
        raw,
        smoothed,
        incumbent,
        np.asarray([True, True, False]),
        policy,
        maximum_incremental_movement_ratio=0.01,
    )
    np.testing.assert_array_equal(gate, [True, False, False])
    np.testing.assert_allclose(
        candidate,
        [6_000.0 + 0.01 * CAPACITY, 6_000.0, 6_000.0],
    )
