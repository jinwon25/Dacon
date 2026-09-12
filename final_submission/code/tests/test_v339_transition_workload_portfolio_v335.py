import numpy as np

from src.archive import v339_transition_workload_portfolio_v335 as v339


def test_add_frozen_increment_preserves_inactive_rows() -> None:
    parent = np.array([0.2, 0.4, 0.6])
    old_parent = np.array([0.1, 0.3, 0.5])
    old_candidate = np.array([0.15, 0.25, 0.7])
    active = np.array([True, False, True])
    result = v339.add_frozen_increment(
        parent, old_parent, old_candidate, active
    )
    assert np.allclose(result, [0.25, 0.4, 0.8])
