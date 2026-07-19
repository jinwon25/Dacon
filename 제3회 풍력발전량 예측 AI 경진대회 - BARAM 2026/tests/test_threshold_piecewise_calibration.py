import numpy as np

from experiments.threshold_piecewise_calibration import CAPACITY, apply_policy


def test_threshold_offset_only_changes_rows_above_gate() -> None:
    prediction = np.asarray([0.09 * CAPACITY, 0.10 * CAPACITY, 0.50 * CAPACITY])
    calibrated = apply_policy(
        prediction,
        {
            "kind": "threshold_offset",
            "minimum_prediction_ratio": 0.10,
            "offset": 575.0,
        },
    )
    assert np.allclose(
        calibrated,
        [0.09 * CAPACITY, 0.10 * CAPACITY + 575.0, 0.50 * CAPACITY + 575.0],
    )


def test_piecewise_offset_uses_forecast_breakpoint_and_clips() -> None:
    prediction = np.asarray([0.39 * CAPACITY, 0.40 * CAPACITY, CAPACITY - 100.0])
    calibrated = apply_policy(
        prediction,
        {
            "kind": "piecewise_offset",
            "breakpoint_ratio": 0.40,
            "low_offset": 450.0,
            "high_offset": 575.0,
        },
    )
    assert np.allclose(
        calibrated,
        [0.39 * CAPACITY + 450.0, 0.40 * CAPACITY + 575.0, CAPACITY],
    )


def test_affine_policy_is_capacity_bounded() -> None:
    prediction = np.asarray([0.0, 5_000.0, CAPACITY])
    calibrated = apply_policy(
        prediction, {"kind": "affine", "scale": 1.05, "offset": 200.0}
    )
    assert np.allclose(calibrated, [200.0, 5_450.0, CAPACITY])
