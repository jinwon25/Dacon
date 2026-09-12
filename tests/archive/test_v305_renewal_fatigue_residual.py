import numpy as np
import pandas as pd

from src.archive.v305_renewal_fatigue_residual import (
    _pooled_dose,
    predict_correction,
    residual_features,
)


def test_residual_features_are_row_local_and_finite() -> None:
    renewal = pd.DataFrame(
        {
            "renewal_expected_age": [10.0, 70.0],
            "renewal_age_sd": [3.0, 8.0],
            "renewal_p_age_ge20": [0.1, 0.95],
            "renewal_p_age_ge60": [0.0, 0.7],
            "renewal_history_median_length": [15.0, 80.0],
            "renewal_history_appearances": [30.0, 20.0],
            "renewal_covered": [1.0, 1.0],
        }
    )
    frame = pd.DataFrame({"inning": [7, 6]})
    features = residual_features(renewal, frame)
    assert features.shape == (2, 11)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[1, "fatigue_age_x_starter"] > 0.0
    assert features.loc[0, "fatigue_age_x_reliever"] > 0.0


def test_pooled_dose_recovers_scalar_projection() -> None:
    direction = np.array([1.0, -2.0, 3.0])
    assert np.isclose(_pooled_dose([(0.25 * direction, direction)]), 0.25)
