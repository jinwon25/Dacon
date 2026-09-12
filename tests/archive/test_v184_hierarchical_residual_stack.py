import numpy as np
import pandas as pd

from src.archive.v184_hierarchical_residual_stack import (
    additive_stack,
    hierarchical_pitcher_features,
)


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [10, 10, 10, 10, 20],
            "asof_pitcher_n": [0, 1, 2, 3, 0],
            "asof_pitcher_success_rate": [0.5, 1.0, 0.5, 2.0 / 3.0, 0.5],
            "asof_pitcher_prev1_game_success_rate": [0.5, 1.0, 0.5, 0.8, 0.5],
            "asof_pitcher_prev3_game_success_rate": [0.5, 1.0, 0.5, 0.7, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.5, 1.0, 0.5, 0.6, 0.5],
        }
    )


def test_hierarchical_features_do_not_use_audit_season_labels() -> None:
    frame = _tiny_frame()
    season = np.array([2019, 2019, 2020, 2020, 2020])
    target = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    changed = target.copy()
    changed[season == 2020] = 1.0 - changed[season == 2020]
    original_features = hierarchical_pitcher_features(frame, target, season)
    changed_features = hierarchical_pitcher_features(frame, changed, season)
    np.testing.assert_allclose(
        original_features.loc[season == 2020].to_numpy(),
        changed_features.loc[season == 2020].to_numpy(),
    )


def test_hierarchical_features_recover_current_season_count() -> None:
    frame = _tiny_frame()
    season = np.array([2019, 2019, 2020, 2020, 2020])
    target = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    features = hierarchical_pitcher_features(frame, target, season)
    # Pitcher 10 ended 2019 with two pitches, so 2020 as-of n=2 and n=3
    # correspond to zero and one current-season pitches.
    np.testing.assert_allclose(
        features.loc[[2, 3], "hier_season_n_log"].to_numpy(),
        np.log1p([0.0, 1.0]),
    )


def test_additive_stack_changes_only_active_local_rows() -> None:
    parent = np.array([0.4, 0.5, 0.6])
    direction = np.array([0.04, -0.08, 0.12])
    local = np.array([0.6, 0.1, 0.2])
    active = np.array([True, False, True])
    result = additive_stack(parent, direction, local, active, 0.01)
    expected = parent + 0.25 * direction
    expected[active] += 0.01 * (local - parent)[active]
    np.testing.assert_allclose(result, expected)
