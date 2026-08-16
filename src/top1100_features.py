"""Fold-safe state/context representation for Top-1100 model families."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_table(history: pd.DataFrame, entity: str, n_col: str, rate_col: str, mix: bool = False) -> pd.DataFrame:
    columns = [entity, "season", n_col, rate_col]
    available = history[columns].copy()
    available["__count"] = np.rint(pd.to_numeric(available[n_col], errors="coerce") * pd.to_numeric(available[rate_col], errors="coerce"))
    last = available.sort_values([entity, "season"]).groupby([entity, "season"], observed=True).tail(1)
    last = last.sort_values([entity, "season"])
    last["__baseline_n"] = last.groupby(entity, observed=True)[n_col].shift(1)
    last["__baseline_s"] = last.groupby(entity, observed=True)["__count"].shift(1)
    # Rename the season-specific previous endpoint; merge_asof handles the
    # strict season inequality without reading current-season rows.
    return last[[entity, "season", "__baseline_n", "__baseline_s"]].dropna(subset=["__baseline_n"]).rename(columns={"season": "__prior_season"})


def _merge_prior(frame: pd.DataFrame, prior: pd.DataFrame, entity: str) -> pd.DataFrame:
    query = frame[[entity, "season"]].copy(); query["__row_order"] = np.arange(len(query))
    # pandas merge_asof requires the asof key to be globally sorted even when
    # a `by` key is supplied.  Sorting season first avoids a subtle failure on
    # multi-entity full-frame builds.
    query = query.sort_values(["season", entity], kind="mergesort")
    prior = prior.sort_values(["__prior_season", entity], kind="mergesort")
    merged = pd.merge_asof(query, prior, left_on="season", right_on="__prior_season", by=entity, allow_exact_matches=False, direction="backward")
    return merged.sort_values("__row_order").reset_index(drop=True)


def build_features(frame: pd.DataFrame, history: pd.DataFrame, *, include_ids: bool = True) -> pd.DataFrame:
    """Build state/context features using only history before each row season."""
    out = pd.DataFrame(index=np.arange(len(frame)))
    base_numeric = ["season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before", "run_total_before", "score_diff_pitcher_team", "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li", "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
    for column in base_numeric:
        if column in frame: out[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float32")
    out["li_log"] = np.log1p(out.get("li", pd.Series(0.0, index=out.index)).clip(lower=0))
    out["close_score"] = (pd.to_numeric(frame.get("score_diff_pitcher_team", 0), errors="coerce").fillna(0).abs() <= 1).astype("int8")
    out["risp"] = ((pd.to_numeric(frame.get("runner_on_2b", 0), errors="coerce").fillna(0) > 0) | (pd.to_numeric(frame.get("runner_on_3b", 0), errors="coerce").fillna(0) > 0)).astype("int8")
    out["late_inning"] = (pd.to_numeric(frame.get("inning", 0), errors="coerce").fillna(0) >= 7).astype("int8")
    out["count_state"] = (frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)).astype("category")
    out["platoon"] = (frame["pitcher_hand"].astype(str) + "_" + frame["batter_hand"].astype(str)).astype("category")
    out["count_platoon"] = (out["count_state"].astype(str) + "_" + out["platoon"].astype(str)).astype("category")
    prior_pitcher = _prior_table(history, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate")
    prior_batter = _prior_table(history, "batter_id", "asof_batter_n", "asof_batter_success_rate")
    pmix = _prior_table(history, "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate")
    p = _merge_prior(frame, prior_pitcher, "pitcher_id"); b = _merge_prior(frame, prior_batter, "batter_id"); m = _merge_prior(frame, pmix, "pitcher_id")
    p_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0); p_count = np.rint(p_n * pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(0))
    b_n = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0); b_count = np.rint(b_n * pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce").fillna(0))
    p0_n = p["__baseline_n"].fillna(0); p0_s = p["__baseline_s"].fillna(0); b0_n = b["__baseline_n"].fillna(0); b0_s = b["__baseline_s"].fillna(0)
    out["pitcher_season_n"] = (p_n - p0_n).clip(lower=0).astype("float32"); out["pitcher_season_success_count"] = (p_count - p0_s).clip(lower=0).astype("float32")
    out["batter_season_n"] = (b_n - b0_n).clip(lower=0).astype("float32"); out["batter_season_success_count"] = (b_count - b0_s).clip(lower=0).astype("float32")
    career_rate = (p_count + 20.0 * 0.52) / (p_n + 20.0); season_rate = (out["pitcher_season_success_count"] + 20.0 * career_rate) / (out["pitcher_season_n"] + 20.0)
    out["pitcher_career_rate"] = career_rate.astype("float32"); out["pitcher_season_rate"] = season_rate.astype("float32"); out["pitcher_season_logit_delta"] = (np.log(np.clip(season_rate, 1e-5, 1-1e-5) / np.clip(1-season_rate, 1e-5, 1)) - np.log(np.clip(career_rate, 1e-5, 1-1e-5) / np.clip(1-career_rate, 1e-5, 1))).astype("float32")
    out["pitcher_state_sd"] = np.sqrt((season_rate * (1-season_rate)) / (out["pitcher_season_n"] + 20.0)).astype("float32"); out["pitcher_history_log_n"] = np.log1p(p_n).astype("float32"); out["batter_history_log_n"] = np.log1p(b_n).astype("float32")
    out["pitcher_newcomer"] = p0_n.eq(0).astype("int8")
    # The last available prior season is strictly before the current season;
    # a gap of two or more seasons indicates a return after an absence.
    out["pitcher_long_gap"] = frame["season"].astype(float).sub(p["__prior_season"].fillna(frame["season"].astype(float))).gt(1).astype("int8")
    # Recent form / physical mix disagreement, all official as-of state.
    for col in ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]:
        if col in frame: out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype("float32")
    out["success_disagreement"] = (out.get("asof_pitcher_prev1_game_success_rate", 0) - out.get("asof_pitcher_prev3_game_success_rate", 0)).abs() + (out.get("asof_pitcher_prev3_game_success_rate", 0) - out.get("asof_pitcher_prev5_game_success_rate", 0)).abs()
    out["middle_disagreement"] = (out.get("asof_pitcher_prev1_game_middle_rate", 0) - out.get("asof_pitcher_prev3_game_middle_rate", 0)).abs() + (out.get("asof_pitcher_prev3_game_middle_rate", 0) - out.get("asof_pitcher_prev5_game_middle_rate", 0)).abs()
    out["pitchmix_entropy"] = -(np.clip(out.get("asof_pitcher_fastball_rate", 0), 1e-8, 1) * np.log(np.clip(out.get("asof_pitcher_fastball_rate", 0), 1e-8, 1)) + np.clip(out.get("asof_pitcher_breaking_rate", 0), 1e-8, 1) * np.log(np.clip(out.get("asof_pitcher_breaking_rate", 0), 1e-8, 1)) + np.clip(out.get("asof_pitcher_offspeed_rate", 0), 1e-8, 1) * np.log(np.clip(out.get("asof_pitcher_offspeed_rate", 0), 1e-8, 1))).astype("float32")
    if include_ids:
        for col in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand", "game_type", "top_bottom", "base_state"]:
            if col in frame: out[col] = frame[col].astype("string").fillna("__MISSING__").astype("category")
    return out
