import numpy as np

from src.archive.v240_brier_classifier_fixed_ensemble_audit import fixed_average


def test_fixed_average_is_exactly_half_and_bounded() -> None:
    actual = fixed_average(np.array([0.2, 1.2]), np.array([0.6, 0.8]))
    np.testing.assert_allclose(actual, [0.4, 0.999])
