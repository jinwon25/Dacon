from __future__ import annotations

import pandas as pd

from src.archive.v231_pressure_specialist_fallback_xgb_oof import (
    restrictions,
    specialist_fit_mask,
)


def test_specialist_mask_is_prior_regular_core_pressure_only() -> None:
    frame = pd.DataFrame({
        "season": [2021, 2021, 2021, 2021, 2022, 2021],
        "game_type": ["R", "R", "R", "F", "R", "R"],
        "pitcher_team_id": [1, 13, 1, 1, 1, 1],
        "batter_team_id": [2, 2, 13, 2, 2, 2],
        "num_runners_on": [1, 1, 1, 1, 1, 0],
        "li": [0.5, 0.5, 0.5, 0.5, 0.5, 1.5],
    })
    assert specialist_fit_mask(frame, 2022).tolist() == [
        True, False, False, False, False, True
    ]


def test_v231_restrictions_isolate_training_population() -> None:
    audit = restrictions()
    assert audit["base_114_features_frozen_from_v217"]
    assert audit["xgb_hyperparameters_frozen_from_v217"]
    assert audit["only_training_population_changed"]
    assert audit["training_population_is_row_local_pressure_rcore"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
