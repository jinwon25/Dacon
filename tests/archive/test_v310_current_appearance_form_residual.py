import numpy as np
import pandas as pd

from src.archive.v310_current_appearance_form_residual import form_residual_features


def test_form_residual_features_are_finite_for_complete_row() -> None:
    appearance = pd.DataFrame({
        "appearance_prev1_excess_z": [1.0],
        "appearance_prev3_excess_z": [0.8],
        "appearance_prev5_excess_z": [0.6],
        "appearance_prev1_rate_k15": [0.55],
        "appearance_prev3_rate_k15": [0.54],
        "appearance_prev5_rate_k15": [0.53],
        "appearance_expected_age": [30.0],
        "appearance_age_sd": [8.0],
        "appearance_typical_length": [80.0],
        "appearance_pitcher_prior": [0.50],
        "appearance_success_est_spread": [1.5],
        "appearance_age_relative_uncertainty": [0.2],
        "appearance_renewal_covered": [1.0],
    })
    frame = pd.DataFrame({"balls_before": [3], "strikes_before": [2], "inning": [7]})
    features = form_residual_features(appearance, frame)
    assert features.shape == (1, 21)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "form_z_x_three_ball"] > 0.0
