import numpy as np

from src.v26_exact_diversity_screen import compose


def test_compose_modes_are_domain_local() -> None:
    parent = np.array([0.40, 0.50, 0.60])
    signal = np.array([0.50, 0.40, 0.70])
    old_base = np.array([0.45, 0.45, 0.55])
    exact_lgb = np.array([0.48, 0.42, 0.65])
    mask = np.array([True, False, True])

    toward = compose(parent, signal, old_base, exact_lgb, mask, 0.2, "toward")
    delta_base = compose(parent, signal, old_base, exact_lgb, mask, 0.2, "delta_base")
    delta_exact = compose(
        parent, signal, old_base, exact_lgb, mask, 0.2, "delta_exact_lgb"
    )

    assert np.allclose(toward, [0.42, 0.50, 0.62])
    assert np.allclose(delta_base, [0.41, 0.50, 0.63])
    assert np.allclose(delta_exact, [0.404, 0.50, 0.61])
    assert toward[1] == parent[1]
