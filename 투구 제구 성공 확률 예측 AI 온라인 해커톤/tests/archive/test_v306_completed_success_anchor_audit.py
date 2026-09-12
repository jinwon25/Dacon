import numpy as np
import pandas as pd

from src.archive.v306_completed_success_anchor_audit import (
    completed_success_anchors,
    patch_success_features,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2],
            "batter_id": [10, 11, 10],
            "asof_pitcher_n": [4, 5, 2],
            "asof_pitcher_success_rate": [0.5, 0.6, 0.5],
            "asof_batter_n": [3, 1, 4],
            "asof_batter_success_rate": [2 / 3, 0.0, 0.5],
            "control_success": [1, 0, 1],
        }
    )


def test_completed_anchor_adds_final_pitch_and_label() -> None:
    anchor = completed_success_anchors(
        _history(), "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    assert anchor[1] == (6.0, 3.0)
    assert anchor[2] == (3.0, 2.0)


def test_patched_opening_success_state_has_zero_delta() -> None:
    history = _history()
    audit = pd.DataFrame(
        {
            "pitcher_id": [1],
            "batter_id": [10],
            "asof_pitcher_n": [6],
            "asof_pitcher_success_rate": [0.5],
            "asof_batter_n": [5],
            "asof_batter_success_rate": [0.6],
            "game_month": [3],
        }
    )
    columns = [
        "p_succ_ssn", "p_succ_ssn_vs_car", "p_succ_ssn_n",
        "p_succ_k25", "p_succ_k75", "p_succ_k400", "p_succ_k1000",
        "b_succ_ssn", "b_succ_ssn_vs_car", "b_succ_ssn_n",
        "b_succ_k25", "b_succ_k75", "b_succ_k400", "b_succ_k1000",
        "p_ppa", "p_est_apps", "p_ssn_per_month",
    ]
    features = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    features["p_ppa"] = 50.0
    patched = patch_success_features(
        audit, history, features, {"p_succ": 0.4, "b_succ": 0.45}
    )
    assert patched.loc[0, "p_succ_ssn_n"] == 0.0
    assert np.isclose(patched.loc[0, "p_succ_ssn"], 0.4)
    assert patched.loc[0, "p_est_apps"] == 0.0
