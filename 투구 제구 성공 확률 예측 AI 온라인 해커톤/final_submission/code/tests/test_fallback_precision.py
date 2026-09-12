import numpy as np
import pandas as pd
from src.archive.v217_rebuild_fallback_xgb_oof import (
    _forward_context_features, situation_masks, TM_COLUMNS,
)


def test_context_differences_round_rates_before_subtracting():
    n = 120
    frame = pd.DataFrame({
        "season": [2022] * 100 + [2023] * 20,
        "control_success": [int(i % 3 == 0) for i in range(n)],
        "pitcher_id": [1] * n, "batter_id": [2] * n,
        "balls_before": [i % 4 for i in range(n)], "strikes_before": [i % 3 for i in range(n)],
        "runner_on_1b": [i % 2 for i in range(n)], "runner_on_2b": [0] * n,
        "runner_on_3b": [0] * n, "batter_hand": [1] * n, "inning": [2] * n,
        "li": [1.] * n, "score_diff_pitcher_team": [0] * n, "game_month": [5] * n,
    })
    output = pd.DataFrame({"p_succ_ssn_n": np.arange(n, dtype=float)})
    tm = pd.DataFrame({"season": [2022, 2022], "pitcher_trackman_id": [11, 11],
                       **{c: [1., 2.] for c in TM_COLUMNS if c != "rel_speed_sd"}})
    mapping = pd.DataFrame({"conf": [1.], "pitcher_id": [1], "pitcher_trackman_id": [11]})
    _forward_context_features(frame, output, tm, mapping)
    prior = frame.iloc[:100]
    raw_overall = prior.control_success.mean()
    mask = situation_masks(prior)["3ball"]
    smoothed = (prior.loc[mask, "control_success"].sum() + 300 * raw_overall) / (mask.sum() + 300)
    expected = np.float32(smoothed) - np.float32(raw_overall)
    assert output.loc[100, "p_sit_3ball_d"] == expected
    expected_role = np.float32(2) * np.log1p(np.float32(100))
    assert output.loc[100, "p_inning_x_role"] == expected_role
