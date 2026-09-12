import numpy as np

from src.archive.v299_count_conditional_dispersion import (
    fit_count_dispersion,
    predict_correction,
)


def test_count_dispersion_protects_group_centers_on_fit_rows() -> None:
    labels = np.array(["0-0"] * 6 + ["0-1"] * 6)
    prediction = np.array(
        [0.40, 0.44, 0.48, 0.52, 0.56, 0.60] * 2, dtype=np.float64
    )
    target = np.array(
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float64
    )
    active = np.ones(len(target), dtype=bool)
    model = fit_count_dispersion(
        labels, prediction, target, active, prior_equivalent_rows=0.0
    )
    correction = predict_correction(model, labels, prediction, active, cap=1.0)
    assert abs(correction[labels == "0-0"].mean()) < 1e-12
    assert abs(correction[labels == "0-1"].mean()) < 1e-12


def test_count_dispersion_is_row_local_and_protects_inactive_rows() -> None:
    labels = np.array(["0-0", "0-0", "0-1", "0-1"])
    prediction = np.array([0.40, 0.60, 0.45, 0.55])
    target = np.array([0.0, 1.0, 1.0, 0.0])
    fit = np.ones(4, dtype=bool)
    model = fit_count_dispersion(
        labels, prediction, target, fit, prior_equivalent_rows=0.0
    )
    active = np.array([True, False, True, False])
    correction = predict_correction(model, labels, prediction, active, cap=1.0)
    assert correction[1] == 0.0
    assert correction[3] == 0.0
    shuffled = np.array([2, 0, 3, 1])
    shuffled_correction = predict_correction(
        model, labels[shuffled], prediction[shuffled], active[shuffled], cap=1.0
    )
    assert np.allclose(shuffled_correction, correction[shuffled])
