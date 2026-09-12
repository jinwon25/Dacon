from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v201_paired_workload_booster import (
    baseline_feature_frame,
    restrictions,
    workload_feature_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "season": [2022, 2022],
            "control_success": [1, 0],
            "top_bottom": ["T", "B"],
            "game_type": ["regular", "regular"],
            "base_state": ["empty", "1b"],
            "pitcher_id": [10, 11],
            "batter_id": [20, 21],
            "pitcher_team_id": [1, 2],
            "batter_team_id": [2, 1],
            "balls_before": [1, 2],
            "asof_pitcher_success_rate": [0.60, 0.55],
            "asof_pitcher_middle_rate": [0.20, 0.25],
            "asof_pitcher_prev1_game_success_rate": [0.50, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.55, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.58, np.nan],
            "asof_pitcher_prev1_game_middle_rate": [0.25, np.nan],
            "asof_pitcher_prev3_game_middle_rate": [0.20, np.nan],
            "asof_pitcher_prev5_game_middle_rate": [0.22, np.nan],
        }
    )


def test_baseline_excludes_labels_season_and_ordinal_ids() -> None:
    features = baseline_feature_frame(_frame())
    for column in (
        "row_id", "season", "control_success", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id",
    ):
        assert column not in features
    assert features["top_bottom_t"].tolist() == [1.0, 0.0]


def test_workload_additions_are_finite_and_missing_neutral() -> None:
    features = workload_feature_frame(_frame())
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "workload_prev1_log_n"] > 0.0
    assert features.loc[1, "workload_prev1_log_n"] == 0.0
    assert features.loc[1, "workload_prev1_missing"] == 1.0
    assert features.loc[1, "workload_prev1_success_reliable_delta"] == 0.0


def test_v201_restrictions_are_forward_and_row_local() -> None:
    audit = restrictions()
    assert audit["official_train_only"]
    assert audit["strictly_prior_season_model_fits"]
    assert audit["paired_identical_booster_configuration"]
    assert audit["row_local_inference"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
