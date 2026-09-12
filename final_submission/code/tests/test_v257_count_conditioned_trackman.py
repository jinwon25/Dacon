import numpy as np
import pandas as pd

from src.archive.v257_count_conditioned_trackman_xgb_oof import (
    METRICS,
    build_count_features,
    make_count_profiles,
)


def _trackman() -> pd.DataFrame:
    rows = []
    for season, season_shift in [(2020, 0.0), (2021, 100.0), (2022, 1000.0)]:
        for balls, count_shift in [(0, 0.0), (3, 30.0)]:
            for pitch in range(4):
                row = {
                    "pitcher_trackman_id": 9,
                    "season": season,
                    "balls_before": balls,
                    "strikes_before": 0,
                    "pitch_type_group": "fastball" if pitch < 3 else "breaking",
                }
                row.update(
                    {
                        metric: season_shift + count_shift + pitch
                        for metric in METRICS
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_count_features_use_latest_strictly_prior_season_and_exact_count() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [3], "pitcher_trackman_id": [9], "conf": [0.99]}
    )
    profiles = make_count_profiles(_trackman(), mapping)
    query = pd.DataFrame(
        {
            "season": [2021, 2022, 2023],
            "pitcher_id": [3, 3, 3],
            "balls_before": [3, 3, 3],
            "strikes_before": [0, 0, 0],
        }
    )
    features = build_count_features(query, profiles)
    np.testing.assert_allclose(
        features["tm_count_mean_rel_speed"], [31.5, 131.5, 1031.5]
    )
    assert (features["tm_count_profile_age"] == 1.0).all()
    assert (features["tm_count_exact_covered"] == 1.0).all()


def test_unseen_count_uses_global_backoff_without_fake_exact_support() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [3], "pitcher_trackman_id": [9], "conf": [0.99]}
    )
    profiles = make_count_profiles(_trackman(), mapping)
    query = pd.DataFrame(
        {
            "season": [2021],
            "pitcher_id": [3],
            "balls_before": [1],
            "strikes_before": [2],
        }
    )
    features = build_count_features(query, profiles)
    assert features.loc[0, "tm_count_exact_covered"] == 0.0
    assert features.loc[0, "tm_count_reliability"] == 0.0
    assert np.isfinite(features.loc[0, "tm_count_mean_rel_speed"])
    assert features.loc[0, "tm_count_reliable_mean_delta_rel_speed"] == 0.0
