from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v228_latest_trackman_fallback_xgb_oof import (
    TM_COLUMNS,
    build_recent_trackman_features,
    restrictions,
)


def test_recent_trackman_features_are_strictly_prior_and_latest() -> None:
    train = pd.DataFrame({
        "season": [2019, 2020, 2021, 2022],
        "pitcher_id": [7, 7, 7, 7],
    })
    rows = []
    for season, speed in [(2019, 140.0), (2020, 144.0), (2021, 143.0)]:
        row = {"pitcher_id": 7, "season": season}
        row.update({column: speed for column in TM_COLUMNS})
        rows.append(row)
    features = build_recent_trackman_features(train, pd.DataFrame(rows))
    assert np.isnan(features.loc[0, "tm_latest_rel_speed"])
    assert features.loc[1, "tm_latest_rel_speed"] == 140.0
    assert features.loc[2, "tm_latest_rel_speed"] == 144.0
    assert features.loc[2, "tm_latest_vs_career_rel_speed"] == 2.0
    assert features.loc[3, "tm_latest_rel_speed"] == 143.0
    assert features.loc[3, "tm_latest_age"] == 1.0
    assert features.loc[3, "tm_latest_covered"] == 1.0


def test_v228_restrictions_isolate_latest_trackman_family() -> None:
    audit = restrictions()
    assert audit["base_114_features_frozen_from_v217"]
    assert audit["xgb_hyperparameters_frozen_from_v217"]
    assert audit["only_latest_trackman_feature_family_added"]
    assert audit["strictly_prior_season_trackman_profiles"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
