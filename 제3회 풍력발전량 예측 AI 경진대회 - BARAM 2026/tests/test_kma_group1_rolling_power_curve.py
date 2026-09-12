from __future__ import annotations

import numpy as np

from experiments.kma_group1_rolling_power_curve import Policy, apply_policy


def test_adaptive_coverage_gate_is_bounded() -> None:
    reference = np.full(20, 10_000.0)
    direct = reference + np.linspace(-5_000.0, 5_000.0, len(reference))
    available = np.ones(len(reference), dtype=bool)
    period = np.ones(len(reference), dtype=bool)
    policy = Policy("both", 0.25, 0.10, 1.00, 0.50)
    candidate, gate, _ = apply_policy(
        reference,
        direct,
        available,
        period,
        policy,
        maximum_movement_ratio=0.05,
    )
    assert gate.sum() == 5
    assert np.max(np.abs(candidate - reference)) <= 0.05 * 21_600.0


def test_direction_gate_moves_only_up_disagreements() -> None:
    reference = np.full(6, 10_000.0)
    direct = reference + np.asarray([-2_000.0, -1_000.0, 1.0, 100.0, 1_000.0, 2_000.0])
    policy = Policy("up", 1.0, 0.10, 1.00, 0.10)
    candidate, gate, _ = apply_policy(
        reference,
        direct,
        np.ones(6, dtype=bool),
        np.ones(6, dtype=bool),
        policy,
        maximum_movement_ratio=0.05,
    )
    assert np.array_equal(gate, np.asarray([False, False, True, True, True, True]))
    assert np.all(candidate[gate] > reference[gate])
