import numpy as np
import pandas as pd

from src.archive.residual import context_features, eb_delta, ridge_fit_predict


def _frame(n=20):
    return pd.DataFrame({
        "balls_before": np.arange(n) % 4,
        "strikes_before": np.arange(n) % 3,
        "pitcher_hand": np.ones(n, dtype=int), "batter_hand": np.ones(n, dtype=int),
        "runner_on_2b": np.zeros(n), "runner_on_3b": np.zeros(n), "base_state": ["0"] * n,
        "outs_before": np.zeros(n), "inning": np.ones(n), "li": np.ones(n), "score_diff_pitcher_team": np.zeros(n),
        "asof_pitcher_n": np.arange(n), "asof_batter_n": np.arange(n),
        "asof_pitcher_prev1_game_success_rate": np.full(n, .5), "asof_pitcher_prev3_game_success_rate": np.full(n, .45), "asof_pitcher_prev5_game_success_rate": np.full(n, .4),
        "asof_pitcher_prev1_game_middle_rate": np.full(n, .3), "asof_pitcher_prev3_game_middle_rate": np.full(n, .35), "asof_pitcher_prev5_game_middle_rate": np.full(n, .4),
        "asof_pitcher_success_rate": np.full(n, .5), "asof_pitcher_middle_rate": np.full(n, .3), "asof_pitcher_ball_rate": np.full(n, .2), "asof_pitcher_strike_rate": np.full(n, .5),
        "asof_pitcher_fastball_rate": np.full(n, .5), "asof_pitcher_breaking_rate": np.full(n, .3), "asof_pitcher_offspeed_rate": np.full(n, .2), "asof_pitcher_pitchmix_n": np.arange(n),
    })


def test_residual_features_exclude_ids_and_bound_eb():
    x = context_features(_frame())
    assert "pitcher_id" not in x.columns and "batter_id" not in x.columns
    delta = eb_delta(x, np.linspace(-.1, .1, len(x)))
    assert np.isfinite(delta).all() and np.max(np.abs(delta)) <= .0125


def test_zero_intercept_ridge_has_finite_output():
    x = context_features(_frame())
    prediction = ridge_fit_predict(x.iloc[:10], np.zeros(10), x.iloc[10:], 1e-3)
    assert np.isfinite(prediction).all()
