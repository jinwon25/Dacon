import numpy as np

from experiments.group2_spatial_gate import (
    GatePolicy,
    apply_policy,
    consensus_gate,
    select_from_development,
)


def test_consensus_gate_requires_direction_uncertainty_and_bounded_distance() -> None:
    base = np.full(5, 10_000.0)
    seed17 = np.asarray([12_000.0, 12_000.0, 12_000.0, 15_000.0, 11_000.0])
    seed29 = np.asarray([12_100.0, 9_000.0, 13_000.0, 15_100.0, 11_100.0])
    policy = GatePolicy(uncertainty_max=0.02, distance_min=0.04, distance_max=0.20, alpha=0.10)

    gate = consensus_gate(base, seed17, seed29, policy)

    assert gate.tolist() == [True, False, False, False, True]


def test_apply_policy_only_changes_gate_and_clips_to_capacity() -> None:
    base = np.asarray([10_000.0, 20_000.0, 10_000.0])
    member = np.asarray([12_000.0, 40_000.0, 2_000.0])
    candidate = apply_policy(base, member, np.asarray([True, True, False]), alpha=1.0)

    np.testing.assert_allclose(candidate, [12_000.0, 21_600.0, 10_000.0])


def test_policy_selection_uses_strict_development_pass_and_robust_floor() -> None:
    records = [
        {
            "policy": {"alpha": 0.10},
            "development_passed": True,
            "development_robust_floor": 0.001,
            "development_mean_score_delta": 0.004,
            "locked_h2": {"delta": -99.0},
        },
        {
            "policy": {"alpha": 0.20},
            "development_passed": True,
            "development_robust_floor": 0.002,
            "development_mean_score_delta": 0.003,
            "locked_h2": {"delta": 99.0},
        },
        {
            "policy": {"alpha": 0.30},
            "development_passed": False,
            "development_robust_floor": 1.0,
            "development_mean_score_delta": 1.0,
        },
    ]

    selected = select_from_development(records)

    assert selected is records[1]
