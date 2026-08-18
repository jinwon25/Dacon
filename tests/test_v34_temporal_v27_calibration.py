import numpy as np
import pandas as pd
import pytest

from src.v34_temporal_v27_calibration import (
    apply_calibrator,
    calibration_correction,
    fit_calibrator,
)


def test_identity_is_exact_when_target_equals_prediction():
    prediction = np.array([0.2, 0.4, 0.6, 0.8])
    domain = np.array(["R_CORE", "R_ANCHOR", "F", "R_CORE"])
    state = fit_calibrator(
        prediction, prediction, domain, family="domain_slope", alpha=100.0
    )
    assert np.allclose(calibration_correction(prediction, domain, state), 0.0)


def test_global_calibration_is_shrunk_toward_identity():
    prediction = np.full(100, 0.4)
    target = np.full(100, 0.6)
    domain = np.full(100, "R_CORE")
    weak = fit_calibrator(target, prediction, domain, family="global", alpha=1.0)
    strong = fit_calibrator(
        target, prediction, domain, family="global", alpha=10_000.0
    )
    weak_shift = calibration_correction(prediction, domain, weak)[0]
    strong_shift = calibration_correction(prediction, domain, strong)[0]
    assert weak_shift > strong_shift > 0.0


def test_apply_calibrator_preserves_protected_domain():
    fit_prediction = np.array([0.4, 0.4])
    state = fit_calibrator(
        np.array([0.6, 0.6]),
        fit_prediction,
        np.array(["R_CORE", "R_CORE"]),
        family="global",
        alpha=1.0,
    )
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.4],
            "v25": [0.4, 0.4],
            "domain3": ["R_CORE", "F"],
        }
    )
    candidate, mask = apply_calibrator(
        frame, state, eta=1.0, apply_domain="R_CORE"
    )
    assert mask.tolist() == [True, False]
    assert candidate[0] > 0.4
    assert candidate[1] == pytest.approx(0.4)
