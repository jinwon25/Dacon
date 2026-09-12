import numpy as np
import pandas as pd

from src.archive.v258_batter_trackman_exposure_xgb_oof import (
    METRICS,
    build_batter_features,
    make_batter_profiles,
)


def _trackman() -> pd.DataFrame:
    rows = []
    for season, shift in [(2020, 0.0), (2021, 100.0), (2022, 1000.0)]:
        for pitch in range(4):
            row = {
                "batter_trackman_id": 8,
                "pitcher_trackman_id": 90 + pitch,
                "season": season,
                "trackman_game_id": f"{season}-{pitch // 2}",
                "balls_before": 3 if pitch == 0 else 0,
                "strikes_before": 2 if pitch == 1 else 0,
                "pitcher_hand": "Left" if pitch % 2 else "Right",
                "pitch_type_group": "fastball" if pitch < 3 else "breaking",
            }
            row.update({metric: shift + pitch for metric in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_batter_features_use_latest_strictly_prior_profile() -> None:
    mapping = pd.DataFrame(
        {
            "batter_id": [4],
            "batter_trackman_id": [8],
            "support": [100],
            "dominance": [1.0],
        }
    )
    profiles = make_batter_profiles(_trackman(), mapping)
    query = pd.DataFrame(
        {"season": [2021, 2022, 2023], "batter_id": [4, 4, 4]}
    )
    features = build_batter_features(query, profiles)
    np.testing.assert_allclose(
        features["bat_tm_latest_mean_rel_speed"], [1.5, 101.5, 1001.5]
    )
    assert (features["bat_tm_profile_age"] == 1.0).all()
    assert (features["bat_tm_profile_covered"] == 1.0).all()


def test_low_support_map_is_not_used() -> None:
    mapping = pd.DataFrame(
        {
            "batter_id": [4],
            "batter_trackman_id": [8],
            "support": [4],
            "dominance": [1.0],
        }
    )
    profiles = make_batter_profiles(_trackman(), mapping)
    assert profiles.empty
