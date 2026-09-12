import numpy as np
import pandas as pd

from src.archive.v44_profile_augmented_pfd import (
    attach_profiles,
    profile_columns,
    student_columns,
)


def test_profile_columns_exclude_keys_and_categorical_link_fields():
    frame = pd.DataFrame(
        columns=[
            "season",
            "pitcher_id",
            "tm_link_confidence_code",
            "tm_link_assignment_rank",
            "tm_speed",
            "tm_spin",
        ]
    )
    assert profile_columns(frame) == ["tm_speed", "tm_spin"]


def test_attach_profiles_preserves_order_and_unknowns():
    rows = pd.DataFrame({"pitcher_id": [2, 1, 3]})
    profiles = pd.DataFrame(
        {
            "season": [2024, 2024],
            "pitcher_id": [1, 2],
            "tm_speed": [140.0, 150.0],
        }
    )
    output = attach_profiles(rows, profiles, 2024, ["tm_speed"])
    np.testing.assert_allclose(output.iloc[:2, 0], [150.0, 140.0])
    assert np.isnan(output.iloc[2, 0])


def test_profile_student_removes_all_identity_columns():
    safe = ["pitcher_id", "batter_id", "pitcher_team_id", "x"]
    assert student_columns(safe, "profile_without_ids") == ["x"]
