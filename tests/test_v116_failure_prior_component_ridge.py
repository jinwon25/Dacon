from __future__ import annotations

import numpy as np

from src.v116_failure_prior_component_ridge import apply_components, fit_ridge


def test_fit_ridge_recovers_unregularized_coefficients() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([0.25, -0.5])
    residual = matrix @ expected
    actual = fit_ridge([matrix], [residual], 0.0, np.zeros(2))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_apply_components_clips_probabilities() -> None:
    parent = np.array([0.2, 0.8])
    matrix = np.array([[1.0], [1.0]])
    actual = apply_components(parent, matrix, np.array([-2.0]))
    np.testing.assert_allclose(actual, np.array([0.001, 0.001]))
