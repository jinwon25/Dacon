import numpy as np

from src.archive.v250_rolling_dual_tree_stacker import (
    LOGISTIC_C,
    logit_features,
    restrictions,
)


def test_logit_features_are_two_column_and_finite() -> None:
    features = logit_features(np.array([0.0, 0.8]), np.array([0.6, 1.0]))
    assert features.shape == (2, 2)
    assert np.isfinite(features).all()


def test_meta_is_regularized_and_strict_forward() -> None:
    assert LOGISTIC_C > 0.0
    audit = restrictions()
    assert audit["meta_training_uses_only_prior_year_oof_and_labels"]
    assert audit["fixed_two_logit_l2_logistic_meta_model"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_fit_or_selection"]
