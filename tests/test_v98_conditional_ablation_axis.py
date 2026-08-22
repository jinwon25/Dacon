import numpy as np

from src.v98_conditional_ablation_axis import _optimal_eta


def test_eta_uses_only_paired_correction():
    axis = {
        "target": np.array([1.0, 0.0]),
        "base_parent": np.array([0.5, 0.5]),
        "parent": np.array([0.5, 0.5]),
        "direct": np.array([0.6, 0.4]),
        "domain3": np.array(["R_CORE", "R_CORE"]),
    }
    assert _optimal_eta([axis], ("R_CORE",), 1.0) == 1.0
