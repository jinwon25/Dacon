from __future__ import annotations

import numpy as np

from scripts.audit_row_region_sequential_gates_brier import brier_gain, row_brier_improvement


def test_brier_gain_matches_direct_unclipped_bss_difference() -> None:
    target = np.array([0.0, 0.0, 1.0, 1.0])
    parent = np.array([0.4, 0.6, 0.4, 0.6])
    candidate = np.array([0.2, 0.4, 0.6, 0.8])
    reference = target.mean() * (1.0 - target.mean())
    expected = 100_000.0 * (
        np.mean(np.square(parent - target))
        - np.mean(np.square(candidate - target))
    ) / reference
    assert brier_gain(target, parent, candidate) == expected


def test_brier_gain_rejects_constant_target() -> None:
    target = np.ones(3)
    prediction = np.full(3, 0.8)
    try:
        brier_gain(target, prediction, prediction)
    except ValueError as error:
        assert "constant target" in str(error)
    else:
        raise AssertionError("constant-target BSS must be rejected")


def test_row_improvement_is_zero_for_protected_rows() -> None:
    target = np.array([0.0, 1.0, 0.0, 1.0])
    parent = np.array([0.3, 0.7, 0.4, 0.6])
    candidate = parent.copy()
    candidate[[1, 3]] += np.array([0.05, 0.10])
    improvement = row_brier_improvement(target, parent, candidate)
    np.testing.assert_array_equal(improvement[[0, 2]], np.zeros(2))
