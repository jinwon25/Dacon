import numpy as np
import pandas as pd

from src.archive.v121_conditional_arsenal_residual import (
    apply_correction,
    arsenal_features,
    build_arsenal_bank,
    fit_ridge,
    predict_ridge,
)


def _history():
    return pd.DataFrame({
        "pitcher_id": [1, 1, 1, 1, 2],
        "batter_hand": [1, 1, 2, 2, 1],
        "count_state": ["0-0", "0-0", "1-0", "1-0", "0-0"],
        "pitch_type_fine": ["Fastball", "Fastball", "Slider", "Slider", "Fastball"],
    })


def test_conditional_arsenal_features_are_row_local_and_unknown_safe():
    bank = build_arsenal_bank(_history(), ["Fastball", "Slider"], 2.0)
    rows = pd.DataFrame({
        "pitcher_id": [1, 999],
        "batter_hand": [1, 2],
        "balls_before": [0, 3],
        "strikes_before": [0, 2],
    })
    features = arsenal_features(rows, bank)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "count_reliability"] > 0.0
    assert features.loc[1, "overall_reliability"] == 0.0
    assert np.isclose(features.loc[1, "count_delta__Fastball"], 0.0)


def test_ridge_and_route_application_have_no_free_intercept():
    x = np.asarray([[1.0], [2.0], [3.0]])
    spec = fit_ridge(x, np.asarray([0.1, 0.2, 0.3]), 0.0)
    np.testing.assert_allclose(predict_ridge(spec, np.asarray([[0.0], [4.0]]), 1.0), [0.0, 0.4])
    candidate = apply_correction(
        np.asarray([0.5, 0.5]), np.asarray(["R_CORE", "F"]),
        np.asarray([0.02, 0.02]), 0.5,
    )
    np.testing.assert_allclose(candidate, [0.51, 0.5])
