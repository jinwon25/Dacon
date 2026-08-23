from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.v79_champion_offset_context import (
    categorical_features,
    fit_coefficients,
    numeric_features,
    rolling_month_splits,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_month": [3, 5, 6, 7, 8, 10],
            "domain3": ["R_CORE", "R_ANCHOR", "F", "R_CORE", "F", "R_ANCHOR"],
            "balls_before": [0, 1, 2, 3, 0, 3],
            "strikes_before": [0, 1, 2, 2, 1, 0],
            "outs_before": [0, 1, 2, 0, 1, 2],
            "num_runners_on": [0, 1, 2, 3, 0, 1],
            "inning": [1, 3, 5, 7, 9, 10],
            "score_diff_pitcher_team": [0, -2, 1, 4, -5, 0],
            "li": [0.5, 1.0, 2.0, 4.0, 0.2, 1.5],
            "home_win_expectancy": [0.5, 0.4, 0.6, 0.7, 0.2, 0.5],
            "away_win_expectancy": [0.5, 0.6, 0.4, 0.3, 0.8, 0.5],
            "pitcher_hand": [1, 1, 2, 2, 1, 2],
            "batter_hand": [1, 2, 1, 2, 2, 1],
            "base_state": ["___", "1__", "_2_", "123", "___", "1_3"],
            "top_bottom": ["T", "B", "T", "B", "T", "B"],
            "asof_pitcher_n": [0, 10, 100, 1000, 20, 200],
            "asof_batter_n": [0, 20, 200, 2000, 30, 300],
            "asof_pitcher_success_rate": [0.5, 0.55, 0.45, 0.6, 0.4, 0.5],
            "asof_batter_success_rate": [0.5, 0.45, 0.55, 0.4, 0.6, 0.5],
            "asof_pitcher_middle_rate": [0.1] * 6,
            "asof_pitcher_reverse_rate": [0.2] * 6,
            "asof_batter_middle_rate": [0.1] * 6,
            "asof_pitcher_prev1_game_success_rate": [0.5] * 6,
            "asof_pitcher_prev3_game_success_rate": [0.5] * 6,
            "asof_pitcher_prev5_game_success_rate": [0.5] * 6,
            "asof_pitcher_prev1_game_middle_rate": [0.1] * 6,
            "asof_pitcher_prev3_game_middle_rate": [0.1] * 6,
            "asof_pitcher_prev5_game_middle_rate": [0.1] * 6,
            "asof_pitcher_fastball_rate": [0.5] * 6,
            "asof_pitcher_breaking_rate": [0.3] * 6,
            "asof_pitcher_offspeed_rate": [0.2] * 6,
            "asof_pitcher_strike_rate": [0.4] * 6,
            "asof_pitcher_ball_rate": [0.3] * 6,
        }
    )


def test_feature_banks_are_finite_and_id_free() -> None:
    numeric = numeric_features(_rows())
    category = categorical_features(_rows())
    assert np.isfinite(numeric.to_numpy()).all()
    prohibited = {"row_id", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    assert prohibited.isdisjoint(numeric.columns)
    assert prohibited.isdisjoint(category.columns)


def test_wayoff_composition_is_bounded() -> None:
    features = numeric_features(_rows())
    assert np.isfinite(features["cmp_wayoff_logit"]).all()
    assert features["cmp_wayoff_logit"].between(-6, 6).all()


def test_rolling_month_splits_are_strictly_forward() -> None:
    rows = pd.concat([_rows()] * 1000, ignore_index=True)
    splits = rolling_month_splits(rows)
    assert len(splits) == 2
    for train, valid in splits:
        assert rows.loc[train, "game_month"].max() < rows.loc[valid, "game_month"].min()


def test_offset_logistic_recovers_a_stable_direction() -> None:
    rng = np.random.default_rng(79)
    x = rng.normal(size=4000)
    matrix = sparse.csr_matrix(x[:, None])
    parent = np.full(len(x), 0.5)
    probability = 1.0 / (1.0 + np.exp(-0.4 * x))
    target = rng.binomial(1, probability)
    coefficients, audit = fit_coefficients(
        matrix,
        target,
        parent,
        np.ones(len(x)),
        l2=0.01,
        maximum_iterations=100,
    )
    assert coefficients[0] > 0.1
    assert audit["feature_count"] == 1
