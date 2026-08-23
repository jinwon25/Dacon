from __future__ import annotations

import numpy as np
import pandas as pd

from src.v78_environment_stable_residual import (
    StableSpec,
    build_features,
    environment_labels,
    predict_correction,
    select_stable_features,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_month": [4, 6, 9],
            "domain3": ["R_CORE", "R_ANCHOR", "F"],
            "pitcher_hand": ["R", "L", "R"],
            "batter_hand": ["R", "R", "L"],
            "asof_pitcher_n": [100, 200, 0],
            "asof_batter_n": [50, 150, 0],
            "asof_pitcher_pitchmix_n": [90, 190, 0],
            "asof_pitcher_success_rate": [0.5, 0.6, 0.5],
            "asof_pitcher_prev1_game_success_rate": [0.6, 0.5, 0.5],
            "asof_pitcher_prev3_game_success_rate": [0.55, 0.55, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.5, 0.6, 0.5],
            "asof_pitcher_prev1_game_middle_rate": [0.1, 0.2, 0.0],
            "asof_pitcher_prev3_game_middle_rate": [0.2, 0.2, 0.0],
            "asof_pitcher_prev5_game_middle_rate": [0.3, 0.1, 0.0],
            "asof_pitcher_strike_rate": [0.3, 0.4, 0.0],
            "asof_pitcher_ball_rate": [0.2, 0.1, 0.0],
            "asof_pitcher_middle_rate": [0.1, 0.2, 0.0],
            "asof_pitcher_reverse_rate": [0.05, 0.1, 0.0],
            "asof_pitcher_fastball_rate": [0.5, 0.4, 1 / 3],
            "asof_pitcher_breaking_rate": [0.3, 0.4, 1 / 3],
            "asof_pitcher_offspeed_rate": [0.2, 0.2, 1 / 3],
            "balls_before": [3, 1, 0],
            "strikes_before": [1, 2, 0],
            "outs_before": [0, 1, 2],
            "num_runners_on": [0, 1, 2],
            "inning": [1, 5, 9],
            "score_diff_pitcher_team": [0, -2, 5],
            "li": [1.0, 2.0, 0.5],
            "home_win_expectancy": [0.5, 0.6, 0.4],
            "away_win_expectancy": [0.5, 0.4, 0.6],
        }
    )


def test_features_are_id_free_finite_and_frozen_width() -> None:
    features = build_features(_rows())
    assert features.shape == (3, 28)
    assert np.isfinite(features.to_numpy()).all()
    prohibited_identifiers = {
        "row_id",
        "pitcher_id",
        "batter_id",
        "game_id",
        "pitch_id",
    }
    assert prohibited_identifiers.isdisjoint(features.columns)


def test_environment_labels_are_target_free_domain_month_bands() -> None:
    assert environment_labels(_rows()).tolist() == [
        "R_CORE|early",
        "R_ANCHOR|mid",
        "F|late",
    ]


def test_stable_selector_keeps_consistent_signal() -> None:
    rng = np.random.default_rng(78)
    environments = np.repeat(np.array(["a", "b", "c", "d"]), 300)
    stable = rng.normal(size=len(environments))
    unstable = rng.normal(size=len(environments))
    signs = np.repeat(np.array([1.0, -1.0, 1.0, -1.0]), 300)
    residual = 0.02 * stable + 0.02 * unstable * signs + rng.normal(
        scale=0.002, size=len(environments)
    )
    selected, audit = select_stable_features(
        np.column_stack([stable, unstable]),
        residual,
        environments,
        ["stable", "unstable"],
        alpha=10.0,
        minimum_environment_rows=100,
        minimum_sign_consistency=0.75,
        minimum_median_abs_coefficient=0.001,
    )
    assert selected.tolist() == [True, False]
    assert audit["selected_features"] == ["stable"]


def test_prediction_uses_only_frozen_spec_and_cap() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    spec = StableSpec(
        feature_names=("x",),
        means=np.array([0.0]),
        scales=np.array([1.0]),
        coefficients=np.array([0.1]),
        risk="uniform",
    )
    assert predict_correction(spec, features, 0.03).tolist() == [0.0, 0.03, 0.03]
