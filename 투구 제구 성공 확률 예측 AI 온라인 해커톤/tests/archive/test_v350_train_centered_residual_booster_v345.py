from __future__ import annotations

import numpy as np

from src.archive.v350_train_centered_residual_booster_v345 import transform_prediction


def test_transform_prediction_uses_frozen_fit_mean_and_dose() -> None:
    raw = np.array([0.01, 0.03, -0.04])
    actual = transform_prediction(raw, fit_prediction_mean=0.01, cap=0.02, dose=0.10)
    np.testing.assert_allclose(actual, np.array([0.0, 0.002, -0.002]))


def test_transform_prediction_is_row_order_independent() -> None:
    raw = np.array([0.01, 0.03, -0.04])
    order = np.array([2, 0, 1])
    expected = transform_prediction(raw, 0.01)[order]
    actual = transform_prediction(raw[order], 0.01)
    np.testing.assert_allclose(actual, expected)
