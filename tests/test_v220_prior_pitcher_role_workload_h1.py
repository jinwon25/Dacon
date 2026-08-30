from __future__ import annotations

import pandas as pd

from src.archive.v220_prior_pitcher_role_workload_h1 import (
    prior_pitcher_role_features,
    restrictions,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2020, 2020, 2020, 2021, 2021],
        "inning": [1, 1, 2, 1, 1],
        "top_bottom": ["T", "T", "B", "T", "T"],
        "pitcher_id": [7, 7, 7, 7, 8],
        "asof_pitcher_success_rate": [0.6] * 5,
        "asof_pitcher_middle_rate": [0.2] * 5,
        "asof_pitcher_prev1_game_success_rate": [0.5] * 5,
        "asof_pitcher_prev1_game_middle_rate": [0.25] * 5,
        "asof_pitcher_prev3_game_success_rate": [0.5] * 5,
        "asof_pitcher_prev3_game_middle_rate": [0.25] * 5,
        "asof_pitcher_prev5_game_success_rate": [0.5] * 5,
        "asof_pitcher_prev5_game_middle_rate": [0.25] * 5,
    })


def test_prior_role_uses_only_earlier_seasons() -> None:
    features = prior_pitcher_role_features(_frame())
    assert features.loc[0, "role_has_history"] == 0.0
    assert features.loc[3, "role_has_history"] == 1.0
    assert features.loc[3, "role_last_median"] == 3.0
    assert features.loc[4, "role_has_history"] == 0.0


def test_prior_role_is_unchanged_by_current_season_row_duplication() -> None:
    frame = _frame()
    base = prior_pitcher_role_features(frame).loc[3, "role_last_median"]
    enlarged = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    repeated = prior_pitcher_role_features(enlarged).loc[3, "role_last_median"]
    assert base == repeated == 3.0


def test_v220_feature_contract_is_target_free() -> None:
    audit = restrictions()
    assert audit["strictly_prior_season_role_stats"]
    assert audit["pitch_counts_from_row_counts_not_target"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
