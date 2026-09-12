from __future__ import annotations

import numpy as np

from src.archive.v225_recent2_fallback_xgb_oof import (
    recent_fit_mask,
    recency_weight,
    restrictions,
)


def test_recent_fit_mask_keeps_only_two_strictly_prior_seasons() -> None:
    season = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    futures = np.zeros(len(season), dtype=bool)
    selected = recent_fit_mask(season, futures, 2024)
    assert season[selected].tolist() == [2022, 2023]


def test_recent_fit_mask_preserves_v217_futures_exclusion() -> None:
    season = np.array([2021, 2021, 2022, 2022, 2023, 2023])
    futures = np.array([False, True, False, True, False, True])
    selected = recent_fit_mask(season, futures, 2023)
    assert selected.tolist() == [True, False, True, False, False, False]
    selected_2024 = recent_fit_mask(season, futures, 2024)
    assert selected_2024.tolist() == [False, False, True, False, True, True]


def test_recency_weight_is_identical_to_v217_formula() -> None:
    weights = recency_weight(np.array([2022, 2023]), 2024)
    np.testing.assert_allclose(weights, [2.0 ** -0.5, 1.0])


def test_v225_restrictions_isolate_the_window_change() -> None:
    audit = restrictions()
    assert audit["feature_contract_frozen_from_v217"]
    assert audit["xgb_hyperparameters_frozen_from_v217"]
    assert audit["only_training_window_changed"]
    assert audit["strictly_prior_season_labels"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
