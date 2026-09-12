import numpy as np
import pandas as pd

from src.archive.v214_joint_workload_h1_screen import joint_workload_features


def test_joint_workload_features_are_row_local_and_lift_compatible_multiple() -> None:
    frame = pd.DataFrame({
        "asof_pitcher_success_rate": [0.50, 0.50],
        "asof_pitcher_middle_rate": [0.20, 0.20],
        "asof_pitcher_prev1_game_success_rate": [0.5, 0.5],
        "asof_pitcher_prev1_game_middle_rate": [0.25, 0.25],
        "asof_pitcher_prev3_game_success_rate": [0.51, 0.51],
        "asof_pitcher_prev3_game_middle_rate": [0.20, 0.20],
        "asof_pitcher_prev5_game_success_rate": [0.505, 0.505],
        "asof_pitcher_prev5_game_middle_rate": [0.20, 0.20],
    })
    full = joint_workload_features(frame)
    singleton = joint_workload_features(frame.iloc[[0]].reset_index(drop=True))
    np.testing.assert_allclose(full.iloc[0].to_numpy(), singleton.iloc[0].to_numpy())
    for horizon in (1, 3, 5):
        assert full.loc[0, f"joint_prev{horizon}_multiplier"] >= 1.0
        assert np.isfinite(full.loc[0, f"joint_prev{horizon}_log_n"])


def test_joint_workload_features_handle_missing_rates() -> None:
    frame = pd.DataFrame({
        "asof_pitcher_success_rate": [np.nan],
        "asof_pitcher_middle_rate": [np.nan],
        **{
            f"asof_pitcher_prev{horizon}_game_{kind}_rate": [np.nan]
            for horizon in (1, 3, 5)
            for kind in ("success", "middle")
        },
    })
    result = joint_workload_features(frame)
    assert np.isfinite(result.to_numpy()).all()
    assert result.loc[0, "joint_valid_horizon_count"] == 0.0
