from __future__ import annotations

import numpy as np

from src.archive.v233_pitcher_fallback_utility_weight_audit import (
    apply_pitcher_weights,
    fit_pitcher_weights,
    restrictions,
)


def test_pitcher_weight_moves_toward_observed_optimum_and_defaults() -> None:
    pitcher = np.array([1, 1, 2])
    parent = np.array([0.50, 0.50, 0.50])
    xgb = np.array([0.60, 0.60, 0.40])
    target = np.array([0.56, 0.56, 0.50])
    active = np.ones(3, dtype=bool)
    weights = fit_pitcher_weights(
        pitcher, target, parent, xgb, active, shrink_rows=0.0
    )
    np.testing.assert_allclose(weights[1], 0.60)
    np.testing.assert_allclose(weights[2], 0.0)
    output, row_weight = apply_pitcher_weights(
        np.array([1, 2, 3]), parent, xgb, active, weights
    )
    np.testing.assert_allclose(row_weight, [0.60, 0.0, 0.30])
    np.testing.assert_allclose(output, [0.56, 0.50, 0.47])


def test_v233_restrictions_are_forward_and_unseen_safe() -> None:
    audit = restrictions()
    assert audit["fallback_xgb_model_frozen"]
    assert audit["pitcher_utility_uses_only_prior_oof_outcomes"]
    assert audit["unseen_pitchers_default_to_public1175_weight"]
    assert audit["source_only_shrink_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
