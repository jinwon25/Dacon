from __future__ import annotations

import numpy as np

from src.archive.v49_temporal_convex_stack import (
    applied_domain_gain,
    bank_matrix,
    fit_simplex,
)


def test_family_matrix_is_deterministic_family_average() -> None:
    names = ["exact::a", "exact::b", "mode::a", "state::a"]
    bank = {
        "exact::a": np.array([0.1, 0.2]),
        "exact::b": np.array([0.3, 0.4]),
        "mode::a": np.array([0.5, 0.6]),
        "state::a": np.array([0.7, 0.8]),
    }
    matrix, columns = bank_matrix(bank, names, "family")
    assert columns == ["exact", "mode", "state"]
    np.testing.assert_allclose(
        matrix,
        np.array([[0.2, 0.5, 0.7], [0.3, 0.6, 0.8]]),
    )


def test_simplex_weights_are_valid_and_prefer_better_column() -> None:
    target = np.array([0.0, 0.0, 1.0, 1.0])
    features = np.column_stack(
        [target, np.full(len(target), 0.5), 1.0 - target]
    )
    weight = fit_simplex(features, target, regularization=0.0)
    assert np.all(weight >= 0.0)
    np.testing.assert_allclose(weight.sum(), 1.0, atol=1e-10)
    assert weight[0] > weight[1]
    assert weight[0] > weight[2]


def test_applied_domain_gain_ignores_unmodified_domain_zeros() -> None:
    result = {
        "domain_gains": {"R_CORE": 4.0, "R_ANCHOR": 0.0, "F": 0.0}
    }
    assert applied_domain_gain(result, "R_CORE") == 4.0
    assert applied_domain_gain(result, "ALL") == 0.0
