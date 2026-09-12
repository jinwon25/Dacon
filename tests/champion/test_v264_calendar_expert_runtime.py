import numpy as np
import pandas as pd

from src.archive.v252_trackman_command_dispersion_xgb_oof import (
    build_forward_features,
)
from src.archive.v258_batter_trackman_exposure_xgb_oof import (
    build_batter_features as build_reference_batter,
)
from src.champion.v264_calendar_expert_runtime import (
    build_batter_features,
    build_command_features,
)


def test_runtime_command_features_match_research_builder() -> None:
    profiles = pd.DataFrame(
        {
            "pitcher_id": [7, 7],
            "season": [2022, 2023],
            "profile_value": [1.0, 3.0],
        }
    )
    rows = pd.DataFrame({"season": [2023, 2024], "pitcher_id": [7, 7]})
    pd.testing.assert_frame_equal(
        build_command_features(rows, profiles),
        build_forward_features(rows, profiles),
    )


def test_runtime_batter_features_match_research_builder() -> None:
    profiles = pd.DataFrame(
        {
            "batter_id": [4, 4],
            "season": [2022, 2023],
            "profile_value": [2.0, 5.0],
        }
    )
    rows = pd.DataFrame({"season": [2023, 2024], "batter_id": [4, 4]})
    expected = build_reference_batter(rows, profiles)
    actual = build_batter_features(rows, profiles)
    pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_allclose(actual["bat_tm_profile_age"], [1.0, 1.0])
