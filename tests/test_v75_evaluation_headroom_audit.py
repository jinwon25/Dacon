from __future__ import annotations

import numpy as np

from src.v75_evaluation_headroom_audit import (
    apply_brier_calibrator,
    fit_brier_calibrator,
)


def test_additive_calibrator_matches_source_mean_residual() -> None:
    target = np.array([0.0, 1.0, 1.0, 0.0])
    parent = np.array([0.2, 0.6, 0.7, 0.1])
    spec = fit_brier_calibrator(target, parent, "additive")
    assert spec["intercept"] == np.mean(target - parent)
    assert np.allclose(apply_brier_calibrator(parent, spec), parent + spec["intercept"])


def test_affine_calibrator_recovers_linear_probability_map() -> None:
    parent = np.linspace(0.1, 0.9, 20)
    target = 0.1 + 0.8 * parent
    spec = fit_brier_calibrator(target, parent, "affine")
    np.testing.assert_allclose(spec["intercept"], 0.1, atol=1e-12)
    np.testing.assert_allclose(spec["slope"], 0.8, atol=1e-12)
