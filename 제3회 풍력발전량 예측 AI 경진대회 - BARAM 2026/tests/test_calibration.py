import numpy as np

from src.calibration import ShrunkResidualCalibrator, optimize_simplex_blend


def test_residual_calibrator_uses_only_prediction_and_lead_at_inference() -> None:
    truth = np.linspace(100, 900, 400)
    pred = truth - 40.0
    lead = np.tile(np.arange(12, 36), 17)[:400]
    calibrator = ShrunkResidualCalibrator(capacity=1_000.0, min_samples=10).fit(truth, pred, lead)

    corrected = calibrator.predict(pred, lead)

    assert corrected.shape == pred.shape
    assert np.all((corrected >= 0) & (corrected <= 1_000))


def test_simplex_blend_is_nonnegative_and_sums_to_one() -> None:
    truth = np.array([200.0, 400.0, 600.0, 800.0])
    predictions = np.column_stack([truth, truth + 100.0])

    weights, _ = optimize_simplex_blend(predictions, truth, 1_000.0, trials=50)

    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[0], 1.0)
