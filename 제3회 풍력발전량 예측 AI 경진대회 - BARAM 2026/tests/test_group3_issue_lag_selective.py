from __future__ import annotations

import numpy as np

from experiments.group3_issue_lag_lgbm import CAPACITY
from experiments.group3_issue_lag_selective import SelectivePolicy, apply_selective


def test_selective_policy_caps_rows_and_blend_weight() -> None:
    base = np.full(100, 0.60 * CAPACITY)
    seed_members = np.vstack(
        [
            base + 0.038 * CAPACITY,
            base + 0.040 * CAPACITY,
            base + 0.042 * CAPACITY,
        ]
    )
    policy = SelectivePolicy(
        max_disagreement=0.06,
        min_base_ratio=0.30,
        max_seed_std_ratio=0.01,
        coverage=0.04,
        alpha=0.05,
        direction="all",
        ranker="absolute_delta",
    )

    candidate, gate = apply_selective(
        base, seed_members, np.ones(100, dtype=bool), policy
    )

    assert gate.sum() == 4
    np.testing.assert_allclose(
        candidate[gate] - base[gate], 0.05 * 0.04 * CAPACITY
    )
    np.testing.assert_array_equal(candidate[~gate], base[~gate])


def test_selective_policy_uses_period_denominator_and_direction() -> None:
    base = np.full(100, 0.60 * CAPACITY)
    seed_members = np.vstack([base + 0.04 * CAPACITY] * 3)
    seed_members[:, 25:50] = base[25:50] - 0.04 * CAPACITY
    period = np.zeros(100, dtype=bool)
    period[:50] = True
    policy = SelectivePolicy(
        max_disagreement=0.06,
        min_base_ratio=0.30,
        max_seed_std_ratio=0.01,
        coverage=0.04,
        alpha=0.02,
        direction="down",
        ranker="signal_to_dispersion",
    )

    _, gate = apply_selective(base, seed_members, period, policy)

    assert gate.sum() == 2
    assert np.all(np.flatnonzero(gate) >= 25)
    assert np.all(np.flatnonzero(gate) < 50)
