import numpy as np
import pandas as pd

from src.archive.v308_tto_fatigue_interaction_residual import (
    tto_residual_features,
)


def test_tto_residual_features_are_finite_for_complete_row() -> None:
    batting = pd.DataFrame(
        {
            "pa_expected_context": [20.0],
            "pa_expected_slot_snap": [19.0],
            "predicted_lineup_slot": [1.0],
            "lineup_history_games": [30.0],
            "context_p_tto2": [0.99],
            "context_p_tto3": [0.75],
            "context_level": [1.0],
            "lineup_covered": [1.0],
        }
    )
    renewal = pd.DataFrame(
        {
            "renewal_expected_age": [65.0],
            "renewal_p_age_ge20": [1.0],
            "renewal_p_age_ge60": [0.8],
            "renewal_history_median_length": [85.0],
            "renewal_covered": [1.0],
        }
    )
    frame = pd.DataFrame(
        {"inning": [6], "balls_before": [3], "strikes_before": [2]}
    )
    features = tto_residual_features(batting, renewal, frame)
    assert features.shape == (1, 27)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "tto_third_x_three_ball"] > 0.0
