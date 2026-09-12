from __future__ import annotations

import numpy as np

from src.archive.v169_independent_residual_gate_audit import apply_direction, optimal_eta


def test_optimal_eta_recovers_known_convex_target() -> None:
    base = np.array([0.2, 0.4, 0.6, 0.8])
    direct = np.array([0.1, 0.5, 0.9, 0.7])
    target = base + 0.25 * (direct - base)
    active = np.ones(4, dtype=bool)
    assert np.isclose(optimal_eta(target, base, direct, active), 0.25)


def test_apply_direction_preserves_inactive_rows() -> None:
    base = np.array([0.2, 0.4, 0.6])
    direct = np.array([0.8, 0.8, 0.8])
    active = np.array([False, True, False])
    output = apply_direction(base, direct, active, 0.5)
    np.testing.assert_array_equal(output[[0, 2]], base[[0, 2]])
    assert np.isclose(output[1], 0.6)
