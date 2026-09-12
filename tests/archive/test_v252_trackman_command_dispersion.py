from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v252_trackman_command_dispersion_xgb_oof import (
    RAW_METRICS,
    build_forward_features,
    make_command_profiles,
)


def _trackman() -> pd.DataFrame:
    rows = []
    for season, side_shift in [(2020, 0.0), (2021, 0.2), (2022, 9.0)]:
        for pitch in range(8):
            row = {
                "pitcher_trackman_id": 11,
                "season": season,
                "game_month": 4 + pitch // 4,
                "trackman_game_id": f"{season}-{pitch // 2}",
                "pitch_type_group": "fastball" if pitch % 2 == 0 else "breaking",
                "rel_speed": 140.0 + pitch,
                "spin_rate": 2200.0 + 3 * pitch,
                "induced_vert_break": 40.0 + pitch,
                "horz_break": 20.0 - pitch,
                "extension": 1.8 + 0.01 * pitch,
                "rel_height": 1.7 + 0.02 * pitch,
                "rel_side": 0.5 + side_shift + 0.01 * pitch,
                "zone_speed": 130.0 + 0.8 * pitch,
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=[
        "pitcher_trackman_id", "season", "game_month", "trackman_game_id",
        "pitch_type_group", *RAW_METRICS,
    ])


def test_profiles_contain_release_and_repeatability_features() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [7], "pitcher_trackman_id": [11], "conf": [0.99]}
    )
    profiles = make_command_profiles(_trackman(), mapping)
    assert len(profiles) == 3
    required = {
        "release_ellipse_scale",
        "release_height_side_corr",
        "within_pitch_sd_rel_side",
        "within_game_sd_rel_height",
        "pitch_type_entropy",
        "monthly_slope_rel_side",
        "sd_deceleration",
    }
    assert required.issubset(profiles.columns)
    assert np.isfinite(profiles["pitch_type_entropy"]).all()


def test_forward_features_never_use_same_or_future_season() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [7], "pitcher_trackman_id": [11], "conf": [0.99]}
    )
    profiles = make_command_profiles(_trackman(), mapping)
    rows = pd.DataFrame(
        {"season": [2021, 2022, 2023, 2023], "pitcher_id": [7, 7, 7, 999]}
    )
    features = build_forward_features(rows, profiles)
    side = "cmd_latest_sd_rel_side"
    expected_2020 = profiles.loc[profiles["season"].eq(2020), "sd_rel_side"].iloc[0]
    expected_2021 = profiles.loc[profiles["season"].eq(2021), "sd_rel_side"].iloc[0]
    expected_2022 = profiles.loc[profiles["season"].eq(2022), "sd_rel_side"].iloc[0]
    assert features.loc[0, side] == np.float32(expected_2020)
    assert features.loc[1, side] == np.float32(expected_2021)
    assert features.loc[2, side] == np.float32(expected_2022)
    assert features.loc[3, "cmd_profile_covered"] == 0.0


def test_low_confidence_mapping_is_excluded() -> None:
    mapping = pd.DataFrame(
        {"pitcher_id": [7], "pitcher_trackman_id": [11], "conf": [0.89]}
    )
    profiles = make_command_profiles(_trackman(), mapping)
    assert profiles.empty
