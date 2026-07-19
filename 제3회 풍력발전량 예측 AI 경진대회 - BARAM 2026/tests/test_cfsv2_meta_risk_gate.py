from __future__ import annotations

import numpy as np

from experiments.cfsv2_meta_risk_gate import _apply_keep_policy


def test_keep_policy_only_retains_high_probability_fine_rows() -> None:
    current = np.array([10.0, 20.0, 30.0, 40.0])
    fine = np.array([11.0, 22.0, 33.0, 44.0])
    gate = np.array([True, True, False, True])
    probability = np.array([0.8, 0.4, 0.9, 0.7])
    candidate, keep = _apply_keep_policy(
        current, fine, gate, probability, threshold=0.65
    )
    assert keep.tolist() == [True, False, False, True]
    assert candidate.tolist() == [11.0, 20.0, 30.0, 44.0]
