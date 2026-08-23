from __future__ import annotations

import numpy as np

from src.v90_trackman_high_pressure_anchor_gate import (
    alternative_gate,
    apply_gate_policy,
    reconstruct_prior,
)


def test_prior_reconstruction_and_policy_replacement() -> None:
    pre = np.asarray([0.4, 0.6, 0.5])
    prior = np.asarray([0.6, 0.4, 0.7])
    gate = np.asarray([0.02, 0.03, 0.0])
    current = pre * (1.0 - gate) + prior * gate
    rebuilt = reconstruct_prior(pre, current, gate)
    assert np.allclose(rebuilt[:2], prior[:2])
    candidate, active, alt = apply_gate_policy(
        pre,
        current,
        gate,
        np.asarray([True, False, True]),
        np.asarray(["R_ANCHOR", "R_ANCHOR", "R_CORE"]),
        "high_only_x100",
    )
    assert active.tolist() == [True, True, False]
    assert np.allclose(alt, [0.02, 0.0, 0.0])
    assert np.isclose(candidate[0], current[0])
    assert np.isclose(candidate[1], pre[1])
    assert np.isclose(candidate[2], current[2])


def test_alternative_gate_is_bounded() -> None:
    gate = np.asarray([0.02, 0.03])
    high = np.asarray([True, False])
    assert np.allclose(alternative_gate(gate, high, "high_only_x150"), [0.03, 0.0])
    assert np.allclose(
        alternative_gate(gate, high, "high_plus_low_half_x100"), [0.02, 0.015]
    )
