import numpy as np

from src.archive.v323_strict_extra_beta_complement import (
    fit_brier_logit_calibration,
    sigmoid,
)


def test_brier_logit_calibration_is_finite_and_direction_preserving():
    probability = np.asarray([0.2, 0.3, 0.7, 0.8] * 20, dtype=float)
    target = np.asarray([0.0, 0.0, 1.0, 1.0] * 20, dtype=float)
    slope, intercept = fit_brier_logit_calibration(target, probability)
    calibrated = sigmoid(slope * np.log(probability / (1.0 - probability)) + intercept)
    assert np.isfinite([slope, intercept]).all()
    assert slope > 0.0
    assert np.all(np.diff(calibrated[:4]) > 0.0)
