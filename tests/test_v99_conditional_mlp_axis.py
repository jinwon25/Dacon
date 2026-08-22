import numpy as np

from src.v99_conditional_mlp_axis import _optimal_eta


def test_optimal_eta_is_capped_and_nonnegative():
    target = np.array([1.0, 0.0])
    parent = np.array([0.5, 0.5])
    assert _optimal_eta(target, parent, np.array([0.1, -0.1]), 0.1) == 0.1
    assert _optimal_eta(target, parent, np.array([-0.1, 0.1]), 0.5) == 0.0
