import numpy as np
import pandas as pd

from src.core.axes import _derived, apply_v22_recipe
from src.v23_structural_residual_screen import _feature_columns


def test_apply_v22_recipe_matches_frozen_additive_formula():
    parent = np.array([0.4, 0.5, 0.6])
    domain = np.array(["R_CORE", "R_ANCHOR", "F"])
    pitcher = np.array([0.6, 0.4, np.nan])
    batter = np.array([0.5, 0.8, np.nan])
    prior = 0.75 * np.nan_to_num(pitcher, nan=0.5) + 0.25 * np.nan_to_num(
        batter, nan=0.5
    )
    expected = parent + np.array(
        [
            0.035 * (0.44 - parent[0]),
            0.020 * (0.48 - parent[1]),
            0.020 * (0.52 - parent[2]),
        ]
    ) + 0.050 * (prior - parent)
    np.testing.assert_allclose(
        apply_v22_recipe(parent, domain, pitcher, batter), expected
    )


def test_context_family_excludes_player_ids():
    frame = pd.DataFrame(
        {
            "row_id": [1],
            "season": [2024],
            "control_success": [1],
            "domain3": ["R_CORE"],
            "pitcher_id": [3],
            "batter_id": [4],
            "game_month": [5],
        }
    )
    columns = _feature_columns(frame, include_ids=False)
    assert "pitcher_id" not in columns
    assert "batter_id" not in columns
    assert columns == ["domain3", "game_month"]


def test_derived_features_are_row_local():
    frame = pd.DataFrame(
        {
            "balls_before": [2],
            "strikes_before": [1],
            "pitcher_hand": ["R"],
            "batter_hand": ["L"],
            "inning": [7],
            "asof_pitcher_n": [99],
            "asof_batter_n": [49],
            "asof_pitcher_pitchmix_n": [24],
            "asof_pitcher_prev1_game_success_rate": [0.6],
            "asof_pitcher_prev5_game_success_rate": [0.5],
            "asof_pitcher_prev1_game_middle_rate": [0.1],
            "asof_pitcher_prev5_game_middle_rate": [0.2],
            "asof_pitcher_success_rate": [0.55],
            "asof_batter_success_rate": [0.50],
        }
    )
    output = _derived(frame, np.array(["R_CORE"]))
    assert output.loc[0, "count_state"] == "2-1"
    assert output.loc[0, "hand_matchup"] == "R-L"
    assert output.loc[0, "inning_bucket"] == "late"
    assert np.isclose(output.loc[0, "pitcher_batter_rate_gap"], 0.05)
