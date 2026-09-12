import numpy as np
import pandas as pd

from src.archive.v316_identity_free_appearance_tree import build_features


def test_features_do_not_require_identity_columns() -> None:
    appearance = pd.DataFrame({
        "appearance_prev1_excess_z": [1.0],
        "appearance_prev3_excess_z": [0.5],
        "appearance_prev5_excess_z": [0.0],
        "appearance_prev1_rate_k15": [0.6],
        "appearance_prev3_rate_k15": [0.55],
        "appearance_prev5_rate_k15": [0.5],
        "appearance_expected_age": [20.0],
        "appearance_age_sd": [5.0],
        "appearance_typical_length": [60.0],
        "appearance_pitcher_prior": [0.5],
        "appearance_success_est_spread": [1.0],
        "appearance_age_relative_uncertainty": [0.2],
        "appearance_renewal_covered": [1.0],
    })
    frame = pd.DataFrame({
        "balls_before": [3], "strikes_before": [2], "inning": [8],
        "outs_before": [2], "num_runners_on": [1],
        "score_diff_pitcher_team": [0], "pitcher_hand": [1], "batter_hand": [1],
    })
    features = build_features(appearance, frame)
    assert np.isfinite(features.to_numpy()).all()
    assert "context_late_close" in features
    assert all("pitcher_id" not in name for name in features.columns)
