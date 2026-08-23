from __future__ import annotations

import numpy as np

from src.archive.v52_failure_aux_logit import (
    failure_targets,
    fit_offset,
    offset_direction,
)


def test_failure_targets_are_disjoint_and_cover_failures() -> None:
    success = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    mode = np.array([0, 1, 2, 3, -1], dtype=np.int8)
    valid, mr, wayoff = failure_targets(success, mode)
    assert np.array_equal(valid, np.array([True, True, True, True, False]))
    assert np.array_equal(mr, np.array([1, 0, 0, 0, 0]))
    assert np.array_equal(wayoff, np.array([0, 1, 0, 0, 0]))


def test_offset_learns_negative_failure_coefficients() -> None:
    target = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    base = np.full(6, 0.5)
    mr = np.array([0.1, 0.2, 0.15, 0.8, 0.7, 0.75])
    wayoff = np.array([0.2, 0.1, 0.15, 0.7, 0.8, 0.75])
    state = fit_offset(target, base, mr, wayoff, l2=0.001)
    assert state.coefficient_mr < 0.0
    assert state.coefficient_wayoff < 0.0


def test_fixed_source_centres_make_mapping_batch_invariant() -> None:
    target = np.array([1, 1, 0, 0], dtype=float)
    base = np.full(4, 0.5)
    mr = np.array([0.1, 0.2, 0.8, 0.7])
    wayoff = np.array([0.2, 0.1, 0.7, 0.8])
    state = fit_offset(target, base, mr, wayoff)
    whole = offset_direction(mr, wayoff, state)
    pieces = np.concatenate(
        [
            offset_direction(mr[:2], wayoff[:2], state),
            offset_direction(mr[2:], wayoff[2:], state),
        ]
    )
    assert np.array_equal(whole, pieces)
