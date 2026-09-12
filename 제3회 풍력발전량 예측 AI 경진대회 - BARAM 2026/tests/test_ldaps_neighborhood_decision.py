from __future__ import annotations

import numpy as np

from experiments.ldaps_neighborhood_decision import (
    CAPACITY,
    DecisionPolicy,
    apply_decision_policy,
    center_neighborhood_scenarios,
    optimize_neighborhood_actions,
)


def test_centered_scenario_median_matches_incumbent() -> None:
    raw = np.asarray(
        [[1_000.0, 2_000.0, 3_000.0], [5_000.0, 6_000.0, 7_000.0]]
    )
    incumbent = np.asarray([8_000.0, 4_000.0])
    centered = center_neighborhood_scenarios(raw, incumbent)
    np.testing.assert_allclose(np.median(centered, axis=1), incumbent)
    np.testing.assert_allclose(
        centered[0] - centered[0, 1], raw[0] - raw[0, 1]
    )


def test_neighborhood_action_never_reduces_expected_scenario_score() -> None:
    scenarios = np.asarray(
        [
            [8_000.0, 8_500.0, 9_000.0, 9_500.0],
            [4_000.0, 4_200.0, 4_500.0, 4_700.0],
        ]
    )
    incumbent = np.asarray([5_000.0, 4_400.0])
    action, before, after = optimize_neighborhood_actions(
        scenarios, incumbent
    )
    assert np.all(after >= before - 1e-12)
    assert action[0] > incumbent[0]


def test_policy_enforces_direction_spread_and_increment_bound() -> None:
    incumbent = np.asarray([5_000.0, 5_000.0, 5_000.0, 5_000.0])
    action = np.asarray([8_000.0, 8_000.0, 2_000.0, 8_000.0])
    spread = np.asarray([500.0, 50.0, 500.0, 500.0])
    gain = np.ones(4)
    available = np.asarray([True, True, True, False])
    policy = DecisionPolicy(
        direction="up",
        coverage=0.10,
        minimum_base_ratio=0.10,
        maximum_base_ratio=0.80,
        minimum_disagreement_kwh=1_000.0,
        spread_mode="high",
        minimum_spread_kwh=100.0,
        maximum_spread_kwh=None,
        alpha=1.0,
    )
    candidate, gate = apply_decision_policy(
        action,
        incumbent,
        spread,
        gain,
        available,
        policy,
        maximum_incremental_movement_ratio=0.01,
    )
    np.testing.assert_array_equal(gate, [True, False, False, False])
    np.testing.assert_allclose(
        candidate, [5_000.0 + 0.01 * CAPACITY, 5_000.0, 5_000.0, 5_000.0]
    )


def test_policy_rejects_nonpositive_expected_gain() -> None:
    incumbent = np.asarray([4_000.0, 4_000.0])
    action = np.asarray([7_000.0, 7_000.0])
    policy = DecisionPolicy(
        direction="both",
        coverage=0.10,
        minimum_base_ratio=0.10,
        maximum_base_ratio=0.80,
        minimum_disagreement_kwh=100.0,
        spread_mode="all",
        minimum_spread_kwh=None,
        maximum_spread_kwh=None,
        alpha=0.5,
    )
    candidate, gate = apply_decision_policy(
        action,
        incumbent,
        np.asarray([500.0, 500.0]),
        np.asarray([0.0, -0.1]),
        np.asarray([True, True]),
        policy,
        maximum_incremental_movement_ratio=0.01,
    )
    assert not gate.any()
    np.testing.assert_allclose(candidate, incumbent)
