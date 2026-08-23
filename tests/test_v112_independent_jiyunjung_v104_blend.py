from __future__ import annotations

import numpy as np

from src.v110_v104_cross_architecture_rebase import apply_correction, fit_alpha


def test_fit_alpha_recovers_bounded_probability_dose() -> None:
    parent = np.array([0.3, 0.4, 0.6, 0.7])
    correction = np.array([-0.1, 0.1, -0.1, 0.1])
    target = parent + 0.2 * correction
    fitted = fit_alpha([target], [parent], [correction], 0.25)
    assert np.isclose(fitted, 0.2)


def test_fit_alpha_keeps_zero_candidate_feasible() -> None:
    parent = np.array([0.2, 0.8])
    correction = np.array([0.1, -0.1])
    target = np.array([0.0, 1.0])
    assert fit_alpha([target], [parent], [correction], 0.25) == 0.0


def test_apply_correction_clips_extreme_probabilities() -> None:
    result = apply_correction(
        np.array([0.99, 0.01]), np.array([0.2, -0.2]), 1.0
    )
    assert np.array_equal(result, np.array([0.999, 0.001]))
