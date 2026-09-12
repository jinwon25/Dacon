import json

import joblib
import numpy as np
import pandas as pd

from src.champion import v285_exact_anchor_fallback_runtime as runtime


def _frame() -> pd.DataFrame:
    values = {
        "row_id": "TEST_1",
        "top_bottom": "T",
        "game_type": "R",
        "base_state": "___",
        "pitcher_hand": 1,
        "batter_hand": 2,
        "pitcher_team_id": 1,
        "batter_team_id": 2,
        "pitcher_id": 10,
        "batter_id": 20,
        "asof_pitcher_n": 120,
        "asof_pitcher_success_rate": 0.5,
        "asof_pitcher_reverse_rate": 0.1,
        "asof_pitcher_middle_rate": 0.2,
        "asof_pitcher_ball_rate": 0.3,
        "asof_pitcher_strike_rate": 0.4,
        "asof_batter_n": 80,
        "asof_batter_success_rate": 0.5,
        "asof_batter_middle_rate": 0.25,
        "asof_pitcher_prev5_game_success_rate": 0.55,
        "asof_pitcher_prev1_game_success_rate": 0.60,
        "balls_before": 0,
        "strikes_before": 0,
        "runner_on_1b": 0,
        "runner_on_2b": 0,
        "runner_on_3b": 0,
        "inning": 1,
        "li": 1.0,
        "score_diff_pitcher_team": 0,
        "game_month": 5,
        "asof_pitcher_pitchmix_n": 0,
    }
    return pd.DataFrame([values])


def _lookup() -> dict:
    prefixes = [spec[3] for spec in runtime.SPECS]
    anchors = {prefix: {10: (100.0, 10.0)} for prefix in prefixes}
    anchors["p_succ"] = {10: (100.0, 40.0)}
    anchors["b_succ"] = {20: (70.0, 30.0)}
    anchors["b_mid"] = {20: (70.0, 15.0)}
    return {
        "features_version": 2,
        "anchor_semantics": "latest_prior_pre_pitch_state",
        "cat": {
            column: {str(_frame().iloc[0][column]): 0} for column in runtime.CAT
        },
        "anchors": anchors,
        "priors": {prefix: 0.4 for prefix in prefixes},
        "overall": {10: 0.5},
        "situations": {name: {10: 0.5} for name in runtime.SITS},
        "pb": {},
        "target_mean": 0.5,
        "ppa": {10: 20.0},
        "ppa_default": 20.0,
        "tm": {column: {} for column in runtime.TM},
    }


def test_exact_runtime_uses_end_anchor_for_multiscale_features(tmp_path) -> None:
    columns = ["p_succ_ssn_n", "p_succ_ssn", "p_succ_k25", "p_est_apps"]
    joblib.dump(_lookup(), tmp_path / "fallback_lookups.joblib")
    (tmp_path / "feature_columns.json").write_text(
        json.dumps(columns), encoding="utf-8"
    )
    features = runtime.build(_frame(), tmp_path)

    np.testing.assert_allclose(features["p_succ_ssn_n"], [20.0])
    np.testing.assert_allclose(features["p_succ_ssn"], [(20.0 + 60.0) / 170.0])
    np.testing.assert_allclose(features["p_succ_k25"], [(20.0 + 10.0) / 45.0])
    np.testing.assert_allclose(features["p_est_apps"], [1.0])


def test_exact_runtime_rejects_stale_lookup(tmp_path) -> None:
    lookup = _lookup()
    lookup["features_version"] = 1
    joblib.dump(lookup, tmp_path / "fallback_lookups.joblib")
    (tmp_path / "feature_columns.json").write_text("[]", encoding="utf-8")
    try:
        runtime.build(_frame(), tmp_path)
    except ValueError as error:
        assert "exact-anchor" in str(error)
    else:
        raise AssertionError("stale lookup was accepted")
