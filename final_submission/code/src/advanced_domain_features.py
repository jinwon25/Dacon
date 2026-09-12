"""Row-local baseball domain features for advanced residual models.

Every value is derived from the current input row or from the corrected
prior-season state produced by :mod:`src.top1100_features`.  The builder does
not inspect frequencies, order, or aggregates of an evaluation batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.corrected_top1100_features import build_features


RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)

LOW_CARDINALITY_COLUMNS = (
    "game_type",
    "top_bottom",
    "base_state",
    "pitcher_hand",
    "batter_hand",
)

TEAM_COLUMNS = ("pitcher_team_id", "batter_team_id")

META_COLUMNS = (
    "base_probability",
    "base_logit",
    "base_uncertainty",
    "base_centered",
)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float32")
    return pd.to_numeric(frame[column], errors="coerce").astype("float32")


def _safe_rate(frame: pd.DataFrame, column: str, fallback: float = 0.52) -> pd.Series:
    return _numeric(frame, column).fillna(fallback).clip(0.0, 1.0).astype("float32")


def build_advanced_domain_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    *,
    include_teams: bool = True,
) -> pd.DataFrame:
    """Build corrected-state and baseball-domain features.

    ``history`` may contain official training rows only.  When ``frame`` is an
    evaluation batch, no statistic is calculated from that batch; each output
    row depends only on its own fields and frozen prior-season endpoints.
    """

    out = build_features(frame, history, include_ids=False).reset_index(drop=True)
    source = frame.reset_index(drop=True)

    for column in LOW_CARDINALITY_COLUMNS:
        out[column] = (
            source[column].astype("string").fillna("__MISSING__").astype("category")
        )
    if include_teams:
        for column in TEAM_COLUMNS:
            out[column] = (
                source[column].astype("string").fillna("__MISSING__").astype("category")
            )
        out["team_matchup"] = (
            out["pitcher_team_id"].astype("string")
            + "_"
            + out["batter_team_id"].astype("string")
        ).astype("category")

    for column in RATE_COLUMNS:
        values = _numeric(source, column)
        out[f"{column}_missing"] = values.isna().astype("int8")
        out[column] = values.fillna(0.52 if "success" in column else 0.0).astype(
            "float32"
        )

    p_success = _safe_rate(source, "asof_pitcher_success_rate")
    b_success = _safe_rate(source, "asof_batter_success_rate")
    p_middle = _safe_rate(source, "asof_pitcher_middle_rate", 0.0)
    b_middle = _safe_rate(source, "asof_batter_middle_rate", 0.0)
    p_reverse = _safe_rate(source, "asof_pitcher_reverse_rate", 0.0)
    p_ball = _safe_rate(source, "asof_pitcher_ball_rate", 0.0)
    p_strike = _safe_rate(source, "asof_pitcher_strike_rate", 0.0)

    success_windows = [
        _safe_rate(source, f"asof_pitcher_prev{window}_game_success_rate")
        for window in (1, 3, 5)
    ]
    middle_windows = [
        _safe_rate(source, f"asof_pitcher_prev{window}_game_middle_rate", 0.0)
        for window in (1, 3, 5)
    ]
    out["recent_success_mean"] = np.mean(success_windows, axis=0).astype("float32")
    out["recent_middle_mean"] = np.mean(middle_windows, axis=0).astype("float32")
    out["recent_success_1_minus_5"] = (success_windows[0] - success_windows[2]).astype(
        "float32"
    )
    out["recent_success_3_minus_5"] = (success_windows[1] - success_windows[2]).astype(
        "float32"
    )
    out["recent_success_3_minus_career"] = (success_windows[1] - p_success).astype(
        "float32"
    )
    out["recent_middle_1_minus_5"] = (middle_windows[0] - middle_windows[2]).astype(
        "float32"
    )
    out["recent_middle_3_minus_career"] = (middle_windows[1] - p_middle).astype(
        "float32"
    )

    out["pitcher_batter_success_gap"] = (p_success - b_success).astype("float32")
    out["pitcher_batter_middle_gap"] = (p_middle - b_middle).astype("float32")
    out["success_minus_strike"] = (p_success - p_strike).astype("float32")
    out["success_minus_ball"] = (p_success - p_ball).astype("float32")
    out["command_risk_sum"] = (p_reverse + p_middle + p_ball).astype("float32")
    out["command_risk_balance"] = (p_middle + p_reverse - p_strike).astype("float32")

    p_n = _numeric(source, "asof_pitcher_n").fillna(0.0).clip(lower=0.0)
    b_n = _numeric(source, "asof_batter_n").fillna(0.0).clip(lower=0.0)
    p_season_n = pd.to_numeric(
        out.get("pitcher_season_n", 0.0), errors="coerce"
    ).fillna(0.0)
    month = _numeric(source, "game_month", 3.0).fillna(3.0)
    elapsed_months = (month - 2.0).clip(lower=1.0)
    out["pitcher_confidence"] = (p_n / (p_n + 200.0)).astype("float32")
    out["batter_confidence"] = (b_n / (b_n + 300.0)).astype("float32")
    out["pitcher_season_confidence"] = (
        p_season_n / (p_season_n + 100.0)
    ).astype("float32")
    out["season_pitch_pace"] = (p_season_n / elapsed_months).astype("float32")
    out["career_pitch_pace"] = (p_n / np.maximum(source["season"] - 2018, 1)).astype(
        "float32"
    )
    out["low_history"] = p_n.lt(100).astype("int8")
    out["high_workload"] = out["season_pitch_pace"].gt(350).astype("int8")

    balls = _numeric(source, "balls_before").fillna(0.0)
    strikes = _numeric(source, "strikes_before").fillna(0.0)
    li = _numeric(source, "li").fillna(0.0).clip(lower=0.0)
    out["count_balance"] = (balls - strikes).astype("float32")
    out["pitches_in_pa_min"] = (balls + strikes).astype("float32")
    out["two_strike"] = strikes.eq(2).astype("int8")
    out["three_ball"] = balls.eq(3).astype("int8")
    out["full_count"] = (balls.eq(3) & strikes.eq(2)).astype("int8")
    out["pitcher_ahead"] = strikes.gt(balls).astype("int8")
    out["hitter_ahead"] = balls.gt(strikes).astype("int8")
    out["count_pressure"] = ((balls + strikes) * np.log1p(li)).astype("float32")
    out["three_ball_pressure"] = (balls.eq(3).astype("float32") * np.log1p(li)).astype(
        "float32"
    )
    out["bases_loaded"] = (
        _numeric(source, "num_runners_on").fillna(0.0).eq(3)
    ).astype("int8")
    score_diff = _numeric(source, "score_diff_pitcher_team").fillna(0.0)
    out["absolute_score_diff"] = score_diff.abs().astype("float32")
    out["tie_game"] = score_diff.eq(0).astype("int8")
    out["win_expectancy_gap"] = (
        _numeric(source, "home_win_expectancy").fillna(50.0)
        - _numeric(source, "away_win_expectancy").fillna(50.0)
    ).astype("float32")

    return out


def add_base_prediction_features(
    features: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    """Append row-local meta features derived from an OOF/base probability."""

    result = features.copy()
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(p) != len(result):
        raise ValueError("base probability length differs from feature rows")
    result["base_probability"] = p.astype("float32")
    result["base_logit"] = np.log(p / (1.0 - p)).astype("float32")
    result["base_uncertainty"] = (p * (1.0 - p)).astype("float32")
    result["base_centered"] = (p - 0.5).astype("float32")
    return result
