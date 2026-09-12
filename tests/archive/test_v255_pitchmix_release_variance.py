import numpy as np
import pandas as pd

from src.archive.v255_pitchmix_release_variance_xgb_oof import (
    METRICS,
    MIX_COLUMNS,
    build_pitchmix_features,
    make_pitch_type_profiles,
)


def _trackman() -> pd.DataFrame:
    rows = []
    for season, shift in [(2020, 0.0), (2021, 10.0), (2022, 100.0)]:
        for group, center in [("fastball", 10.0), ("breaking", 20.0), ("offspeed", 30.0)]:
            for pitch in range(4):
                value = center + shift + pitch
                row = {
                    "pitcher_trackman_id": 9, "season": season,
                    "pitch_type_group": group,
                }
                row.update({metric: value for metric in METRICS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_mix_weighted_features_use_latest_strictly_prior_profile() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [3], "pitcher_trackman_id": [9], "conf": [0.99]}
    )
    profiles = make_pitch_type_profiles(_trackman(), mapping)
    rows = pd.DataFrame({
        "season": [2021, 2022, 2023], "pitcher_id": [3, 3, 3],
        MIX_COLUMNS[0]: [1.0, 1.0, 1.0],
        MIX_COLUMNS[1]: [0.0, 0.0, 0.0],
        MIX_COLUMNS[2]: [0.0, 0.0, 0.0],
    })
    features = build_pitchmix_features(rows, profiles)
    # Fastball means are 11.5, 21.5 and 111.5 in seasons 2020, 2021 and 2022.
    np.testing.assert_allclose(
        features["mix_expected_mean_rel_speed"], [11.5, 21.5, 111.5]
    )
    assert (features["mix_profile_age"] == 1.0).all()


def test_mix_total_variance_includes_between_pitch_type_separation() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [3], "pitcher_trackman_id": [9], "conf": [0.99]}
    )
    profiles = make_pitch_type_profiles(_trackman(), mapping)
    rows = pd.DataFrame({
        "season": [2021], "pitcher_id": [3],
        MIX_COLUMNS[0]: [0.5], MIX_COLUMNS[1]: [0.5], MIX_COLUMNS[2]: [0.0],
    })
    features = build_pitchmix_features(rows, profiles)
    assert features.loc[0, "mix_total_sd_rel_side"] > features.loc[
        0, "mix_within_sd_rel_side"
    ]
