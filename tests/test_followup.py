import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.archive.calibration import BetaCalibrator, InterceptCalibrator, fit_constrained_blend
from src.archive.features import FeatureBuilder
from src.metrics import cluster_bootstrap_delta
from src.archive.validation import historical_oof_indices, walk_forward_splits
from tests.test_features import _main_rows


def test_walk_forward_outer_folds_are_strictly_past_only():
    frame = pd.DataFrame(
        {
            "season": [2019, 2020, 2021, 2022, 2023, 2024],
            "control_success": [0, 1, 0, 1, 0, 1],
        }
    )
    folds = walk_forward_splits(frame, (2021, 2022, 2023, 2024))
    assert [fold.validation_season for fold in folds] == [2021, 2022, 2023, 2024]
    for fold in folds:
        assert frame.iloc[fold.train_idx]["season"].max() < fold.validation_season
        assert set(frame.iloc[fold.valid_idx]["season"]) == {fold.validation_season}


def test_nested_oof_never_contains_outer_or_future_season():
    frame = pd.DataFrame(
        {
            "season": [2019, 2020, 2021, 2022, 2023, 2024],
            "control_success": [0, 1, 0, 1, 0, 1],
        }
    )
    folds = historical_oof_indices(frame, outer_validation_season=2024)
    assert folds
    assert max(fold.validation_season for fold in folds) == 2023
    for fold in folds:
        assert frame.iloc[fold.train_idx]["season"].max() < fold.validation_season < 2024


def test_beta_identity_map_is_available_before_fit():
    probability = np.array([0.01, 0.2, 0.5, 0.9, 0.99])
    actual = BetaCalibrator().predict(probability)
    np.testing.assert_allclose(actual, probability, atol=1e-12)


def test_calibration_fit_and_apply_are_separate():
    calibration_prediction = np.array([0.1, 0.2, 0.7, 0.8])
    calibration_target = np.array([0, 0, 1, 1])
    untouched_validation = np.array([0.25, 0.75])
    calibrator = InterceptCalibrator().fit(calibration_prediction, calibration_target)
    first = calibrator.predict(untouched_validation)
    second = calibrator.predict(untouched_validation)
    np.testing.assert_allclose(first, second)


def test_constrained_blend_is_nonnegative_and_sum_one():
    target = np.array([0, 0, 1, 1], dtype=float)
    first = np.array([0.1, 0.3, 0.7, 0.9])
    second = np.array([0.2, 0.2, 0.8, 0.8])
    weight, score = fit_constrained_blend([first, second], target, l2=1e-3)
    assert np.all(weight >= 0)
    assert np.isclose(weight.sum(), 1.0)
    assert np.isfinite(score)


def test_hierarchical_v2_is_row_local_and_drops_season():
    rows = _main_rows()
    builder = FeatureBuilder("hierarchical_v2", drop_columns=["season"]).fit(
        rows, np.array([0, 1])
    )
    together = builder.transform(rows).iloc[[0]].reset_index(drop=True)
    alone = builder.transform(rows.iloc[[0]]).reset_index(drop=True)
    assert "season" not in together.columns
    assert "pitcher_posterior_se" in together.columns
    assert_frame_equal(together, alone)


def test_cluster_bootstrap_detects_uniform_improvement():
    target = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    incumbent = np.full(6, 0.5)
    candidate = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    result = cluster_bootstrap_delta(
        target, candidate, incumbent, clusters, n_resamples=200, seed=1
    )
    assert result["observed_delta"] < 0
    assert result["improvement_probability"] == 1.0
    assert result["n_improving_resamples"] == result["n_resamples"]
    assert 0.0 < result["mc_probability_ci_lower"] < 1.0
    assert result["mc_probability_ci_upper"] == 1.0
