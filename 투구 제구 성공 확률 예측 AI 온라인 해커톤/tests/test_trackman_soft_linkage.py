import numpy as np

from src.trackman_soft_linkage import bootstrap_soft_linkage


def test_bootstrap_soft_linkage_is_one_to_one_and_reproducible():
    cost = np.array([[0.01, 0.2, 0.3], [0.2, 0.02, 0.3], [0.3, 0.2, 0.03]])
    first = bootstrap_soft_linkage(cost, repeats=20, seed=42)
    second = bootstrap_soft_linkage(cost, repeats=20, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.allclose(first[0].sum(axis=1), 1.0)
    assert np.all(first[1] >= 0.0)
    assert np.all(first[1] <= 1.0)
