from __future__ import annotations

import numpy as np

from src.archive.v293_futures_population_lgb_diversity import F_WEIGHT, _compose


def test_same_total_mean_has_fixed_total_dose() -> None:
    base = np.array([0.4, 0.6, 0.5])
    parent = np.array([0.42, 0.6, 0.48])
    f_mask = np.array([True, False, True])
    cat = np.array([0.6, 0.3])
    lgb = np.array([0.5, 0.4])
    actual, active = _compose("same_total_mean", base, parent, f_mask, cat, lgb)
    expected = parent.copy()
    expected[f_mask] = base[f_mask] + F_WEIGHT * (
        0.5 * (cat - base[f_mask]) + 0.5 * (lgb - base[f_mask])
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(active, f_mask)


def test_consensus_add_changes_only_direction_agreement() -> None:
    base = np.array([0.4, 0.6, 0.5])
    parent = np.array([0.42, 0.6, 0.48])
    f_mask = np.array([True, False, True])
    cat = np.array([0.6, 0.4])
    lgb = np.array([0.5, 0.6])
    actual, active = _compose("consensus_add", base, parent, f_mask, cat, lgb)
    np.testing.assert_array_equal(active, [True, False, False])
    assert actual[1] == parent[1]
    assert actual[2] == parent[2]
