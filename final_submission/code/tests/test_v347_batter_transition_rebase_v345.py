from __future__ import annotations

import numpy as np

from src.archive.v347_batter_transition_rebase_v345 import full_row_rms


def test_full_row_rms_is_order_invariant() -> None:
    parent = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    candidate = np.array([0.3, 0.4, 0.8], dtype=np.float64)
    order = np.array([2, 0, 1])
    assert full_row_rms(parent, candidate) == full_row_rms(parent[order], candidate[order])


def test_full_row_rms_zero_for_parity() -> None:
    parent = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    assert full_row_rms(parent, parent) == 0.0
