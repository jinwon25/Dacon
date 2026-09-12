import numpy as np

from src.trackman_privileged_distillation import bss


def test_bss_improves_for_better_probability_forecast():
    target = np.array([0.0, 0.0, 1.0, 1.0])
    weak = np.array([0.5, 0.5, 0.5, 0.5])
    strong = np.array([0.1, 0.2, 0.8, 0.9])
    assert bss(target, strong) > bss(target, weak)
