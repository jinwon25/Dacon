"""Offline evaluation entry point.

Reads ./data/test.csv relative to this file and writes
./output/submission.csv. Every feature is row-local or uses training artifacts
inside ./model; no test-set aggregate, frequency, order, or distribution is
used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_PATH = BASE_DIR / "data" / "test.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__")


def hierarchical_prior(
    frame: pd.DataFrame,
    global_rate: float,
    alpha: float,
    batter_weight: float,
) -> np.ndarray:
    pn = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(float)
    bn = (
        pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0).to_numpy(float)
        * batter_weight
    )
    pr = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(global_rate)
        .to_numpy(float)
    )
    br = (
        pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce")
        .fillna(global_rate)
        .to_numpy(float)
    )
    return (pn * pr + bn * br + alpha * global_rate) / (pn + bn + alpha)


def build_features(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Mirror src.features.FeatureBuilder using only frozen train artifacts."""
    base_columns = list(spec["base_columns"])
    missing = [column for column in base_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"test columns missing: {missing}")
    out = frame[base_columns].copy()
    feature_set = spec["feature_set"]
    if feature_set == "no_asof":
        out = out.drop(columns=[column for column in out if column.startswith("asof_")])
    if feature_set in {"engineered", "trackman", "trackman_pitcher"}:
        count = _safe_string(out["balls_before"]) + "-" + _safe_string(out["strikes_before"])
        out["count_state"] = count
        out["platoon"] = _safe_string(out["pitcher_hand"]) + "-" + _safe_string(out["batter_hand"])
        out["inning_half"] = _safe_string(out["inning"]) + "-" + _safe_string(out["top_bottom"])
        out["pitcher_count"] = _safe_string(out["pitcher_id"]) + "-" + count
        out["pitcher_batter_hand"] = _safe_string(out["pitcher_id"]) + "-" + _safe_string(out["batter_hand"])
        out["batter_count"] = _safe_string(out["batter_id"]) + "-" + count
        out["team_matchup"] = _safe_string(out["pitcher_team_id"]) + "-" + _safe_string(out["batter_team_id"])
        out["situation_state"] = count + "-" + _safe_string(out["outs_before"]) + "-" + _safe_string(out["base_state"])
        out["abs_score_diff"] = pd.to_numeric(out["score_diff_pitcher_team"]).abs().astype("float32")
        out["late_inning"] = (pd.to_numeric(out["inning"]) >= 7).astype("int8")
        out["runners_scoring_position"] = (
            (pd.to_numeric(out["runner_on_2b"]) > 0)
            | (pd.to_numeric(out["runner_on_3b"]) > 0)
        ).astype("int8")
        out["pitcher_log_n"] = np.log1p(pd.to_numeric(out["asof_pitcher_n"])).astype("float32")
        out["batter_log_n"] = np.log1p(pd.to_numeric(out["asof_batter_n"])).astype("float32")
        out["pitchmix_log_n"] = np.log1p(pd.to_numeric(out["asof_pitcher_pitchmix_n"])).astype("float32")
        rate = float(spec["global_rate"])
        pn = pd.to_numeric(out["asof_pitcher_n"], errors="coerce").astype("float64")
        bn = pd.to_numeric(out["asof_batter_n"], errors="coerce").astype("float64")
        pr = pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce").fillna(rate)
        br = pd.to_numeric(out["asof_batter_success_rate"], errors="coerce").fillna(rate)
        for alpha in (50.0, 200.0):
            out[f"pitcher_success_smooth_{int(alpha)}"] = (
                (pn * pr + alpha * rate) / (pn + alpha)
            ).astype("float32")
        out["batter_success_smooth_200"] = ((bn * br + 200.0 * rate) / (bn + 200.0)).astype("float32")
        out["hierarchical_success_prior"] = hierarchical_prior(out, rate, 200.0, 0.25).astype("float32")
        out["pitcher_recent_delta_1"] = (
            pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"])
            - out["pitcher_success_smooth_200"]
        ).astype("float32")
        out["pitcher_recent_delta_3"] = (
            pd.to_numeric(out["asof_pitcher_prev3_game_success_rate"])
            - out["pitcher_success_smooth_200"]
        ).astype("float32")
        out["pitcher_recent_delta_5"] = (
            pd.to_numeric(out["asof_pitcher_prev5_game_success_rate"])
            - out["pitcher_success_smooth_200"]
        ).astype("float32")
    if feature_set == "trackman":
        context_path = MODEL_DIR / "trackman_context.csv"
        context = pd.read_csv(context_path)
        out["__row_order"] = np.arange(len(out))
        out = out.merge(
            context,
            on=["season", "balls_before", "strikes_before", "outs_before"],
            how="left",
            sort=False,
            validate="many_to_one",
        ).sort_values("__row_order", kind="stable")
        out = out.drop(columns="__row_order").reset_index(drop=True)
        out["tm_context_missing"] = out["tm_context_n"].isna().astype("int8")
    if feature_set == "trackman_pitcher":
        profile = pd.read_csv(MODEL_DIR / "trackman_pitcher_profiles.csv")
        out["__row_order"] = np.arange(len(out))
        out = out.merge(
            profile,
            on=["season", "pitcher_id"],
            how="left",
            sort=False,
            validate="many_to_one",
        ).sort_values("__row_order", kind="stable")
        out = out.drop(columns="__row_order").reset_index(drop=True)
        out["tm_pitcher_profile_missing"] = (
            out["tm_linked"].isna() | (out["tm_linked"] <= 0)
        ).astype("int8")

    category_maps = spec["category_maps"]
    for column in spec["categorical_columns"]:
        mapping = {str(key): int(value) for key, value in category_maps[column].items()}
        out[column] = _safe_string(out[column]).map(mapping).fillna(-1).astype("int32")
    for column in out.columns:
        if column not in spec["categorical_columns"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out


def apply_calibration(probability: np.ndarray, calibration: dict, global_rate: float) -> np.ndarray:
    method = calibration["method"]
    if method == "identity":
        return probability
    if method == "base_rate_shrink":
        weight = float(calibration["weight"])
        return (1.0 - weight) * probability + weight * global_rate
    if method == "logit_offset":
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)) + float(calibration["offset"])
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    if method == "platt":
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        value = float(calibration["coefficient"]) * logits + float(calibration["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
    if method == "isotonic":
        return np.interp(
            probability,
            np.asarray(calibration["x_thresholds"], dtype=float),
            np.asarray(calibration["y_thresholds"], dtype=float),
        )
    raise ValueError(f"unknown calibration method: {method}")


def apply_optional_game_type_offsets(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply frozen train-only offsets; each row depends only on its game_type."""
    artifact = MODEL_DIR / "game_type_offsets.json"
    if not artifact.exists():
        return probability
    spec = json.loads(artifact.read_text(encoding="utf-8"))
    if spec.get("method") != "game_type_logit_offset":
        raise ValueError("unsupported game_type offset artifact")
    if "game_type" not in frame.columns:
        raise ValueError("test is missing game_type")
    output = np.asarray(probability, dtype=np.float64).copy()
    game_type = frame["game_type"].astype("string").fillna("__MISSING__")
    for value, raw_offset in spec["offsets"].items():
        offset = float(raw_offset)
        mask = game_type.eq(str(value)).to_numpy()
        if mask.any() and offset != 0.0:
            clipped = np.clip(output[mask], 1e-6, 1.0 - 1e-6)
            logit = np.log(clipped / (1.0 - clipped)) + offset
            output[mask] = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    return output


def _frozen_endpoint(frame: pd.DataFrame, entity: str, filename: str) -> pd.DataFrame:
    """Map a train-only entity endpoint onto rows without batch statistics."""
    endpoint = pd.read_csv(MODEL_DIR / filename, dtype={entity: "string"})
    if endpoint[entity].isna().any() or endpoint[entity].duplicated().any():
        raise ValueError(f"invalid frozen endpoint table: {filename}")
    endpoint = endpoint.set_index(entity)
    keys = frame[entity].astype("string")
    return pd.DataFrame(
        {
            "__prior_season": keys.map(endpoint["__prior_season"]),
            "__baseline_n": keys.map(endpoint["__baseline_n"]),
            "__baseline_s": keys.map(endpoint["__baseline_s"]),
        },
        index=np.arange(len(frame)),
    )


def build_corrected_state_features(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Reproduce the corrected 2025 state features from frozen train endpoints."""
    out = pd.DataFrame(index=np.arange(len(frame)))
    base_numeric = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
        "run_total_before",
        "score_diff_pitcher_team",
        "num_runners_on",
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_pitchmix_n",
    ]
    missing = [column for column in base_numeric if column not in frame.columns]
    if missing:
        raise ValueError(f"corrected-state input columns missing: {missing}")
    for column in base_numeric:
        out[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0.0)
            .astype("float32")
            .to_numpy()
        )

    out["li_log"] = np.log1p(out["li"].clip(lower=0))
    score_diff = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="coerce"
    ).fillna(0)
    out["close_score"] = score_diff.abs().le(1).astype("int8").to_numpy()
    runner_2b = pd.to_numeric(frame["runner_on_2b"], errors="coerce").fillna(0)
    runner_3b = pd.to_numeric(frame["runner_on_3b"], errors="coerce").fillna(0)
    out["risp"] = (runner_2b.gt(0) | runner_3b.gt(0)).astype("int8").to_numpy()
    out["late_inning"] = (
        pd.to_numeric(frame["inning"], errors="coerce")
        .fillna(0)
        .ge(7)
        .astype("int8")
        .to_numpy()
    )
    count_state = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    platoon = frame["pitcher_hand"].astype(str) + "_" + frame["batter_hand"].astype(str)
    out["count_state"] = count_state.to_numpy()
    out["platoon"] = platoon.to_numpy()
    out["count_platoon"] = (count_state + "_" + platoon).to_numpy()

    pitcher = _frozen_endpoint(
        frame, "pitcher_id", "corrected_state_pitcher_prior.csv"
    )
    batter = _frozen_endpoint(
        frame, "batter_id", "corrected_state_batter_prior.csv"
    )
    pitcher_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0)
    pitcher_count = np.rint(
        pitcher_n
        * pd.to_numeric(
            frame["asof_pitcher_success_rate"], errors="coerce"
        ).fillna(0)
    )
    batter_n = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0)
    batter_count = np.rint(
        batter_n
        * pd.to_numeric(
            frame["asof_batter_success_rate"], errors="coerce"
        ).fillna(0)
    )
    pitcher_prior_n = pitcher["__baseline_n"].fillna(0)
    pitcher_prior_s = pitcher["__baseline_s"].fillna(0)
    batter_prior_n = batter["__baseline_n"].fillna(0)
    batter_prior_s = batter["__baseline_s"].fillna(0)

    out["pitcher_season_n"] = (
        pitcher_n.to_numpy() - pitcher_prior_n.to_numpy()
    ).clip(min=0).astype("float32")
    out["pitcher_season_success_count"] = (
        np.asarray(pitcher_count) - pitcher_prior_s.to_numpy()
    ).clip(min=0).astype("float32")
    out["batter_season_n"] = (
        batter_n.to_numpy() - batter_prior_n.to_numpy()
    ).clip(min=0).astype("float32")
    out["batter_season_success_count"] = (
        np.asarray(batter_count) - batter_prior_s.to_numpy()
    ).clip(min=0).astype("float32")

    career_rate = (np.asarray(pitcher_count) + 20.0 * 0.52) / (
        pitcher_n.to_numpy() + 20.0
    )
    prior_rate = (pitcher_prior_s.to_numpy() + 20.0 * 0.52) / (
        pitcher_prior_n.to_numpy() + 20.0
    )
    season_rate = (
        out["pitcher_season_success_count"].to_numpy() + 20.0 * prior_rate
    ) / (out["pitcher_season_n"].to_numpy() + 20.0)
    batter_prior_rate = (batter_prior_s.to_numpy() + 20.0 * 0.52) / (
        batter_prior_n.to_numpy() + 20.0
    )
    batter_season_rate = (
        out["batter_season_success_count"].to_numpy()
        + 20.0 * batter_prior_rate
    ) / (out["batter_season_n"].to_numpy() + 20.0)
    out["pitcher_career_rate"] = career_rate.astype("float32")
    out["pitcher_prior_rate"] = prior_rate.astype("float32")
    out["pitcher_season_rate"] = season_rate.astype("float32")
    season_clipped = np.clip(season_rate, 1e-5, 1.0 - 1e-5)
    prior_clipped = np.clip(prior_rate, 1e-5, 1.0 - 1e-5)
    out["pitcher_season_logit_delta"] = (
        np.log(season_clipped / np.clip(1.0 - season_rate, 1e-5, 1.0))
        - np.log(prior_clipped / np.clip(1.0 - prior_rate, 1e-5, 1.0))
    ).astype("float32")
    out["batter_prior_rate"] = batter_prior_rate.astype("float32")
    out["batter_season_rate"] = batter_season_rate.astype("float32")
    out["pitcher_state_sd"] = np.sqrt(
        (season_rate * (1.0 - season_rate))
        / (out["pitcher_season_n"].to_numpy() + 20.0)
    ).astype("float32")
    out["pitcher_history_log_n"] = np.log1p(pitcher_n.to_numpy()).astype("float32")
    out["batter_history_log_n"] = np.log1p(batter_n.to_numpy()).astype("float32")
    out["pitcher_newcomer"] = pitcher_prior_n.eq(0).astype("int8").to_numpy()
    current_season = pd.to_numeric(frame["season"], errors="coerce").to_numpy(float)
    prior_season = pitcher["__prior_season"].to_numpy(float)
    prior_season = np.where(np.isnan(prior_season), current_season, prior_season)
    out["pitcher_long_gap"] = ((current_season - prior_season) > 1).astype("int8")

    asof_columns = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    for column in asof_columns:
        out[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0.0)
            .astype("float32")
            .to_numpy()
        )
    out["success_disagreement"] = (
        (out["asof_pitcher_prev1_game_success_rate"] - out["asof_pitcher_prev3_game_success_rate"]).abs()
        + (out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_prev5_game_success_rate"]).abs()
    )
    out["middle_disagreement"] = (
        (out["asof_pitcher_prev1_game_middle_rate"] - out["asof_pitcher_prev3_game_middle_rate"]).abs()
        + (out["asof_pitcher_prev3_game_middle_rate"] - out["asof_pitcher_prev5_game_middle_rate"]).abs()
    )
    fastball = np.clip(out["asof_pitcher_fastball_rate"], 1e-8, 1.0)
    breaking = np.clip(out["asof_pitcher_breaking_rate"], 1e-8, 1.0)
    offspeed = np.clip(out["asof_pitcher_offspeed_rate"], 1e-8, 1.0)
    out["pitchmix_entropy"] = -(
        fastball * np.log(fastball)
        + breaking * np.log(breaking)
        + offspeed * np.log(offspeed)
    ).astype("float32")

    for column in ("game_type", "top_bottom", "base_state", "pitcher_hand", "batter_hand"):
        out[column] = _safe_string(frame[column]).to_numpy()
    missing_features = [column for column in spec["feature_columns"] if column not in out]
    if missing_features:
        raise ValueError(f"corrected-state features missing: {missing_features}")
    out = out[spec["feature_columns"]].copy()
    for column in spec["categorical_columns"]:
        values = out[column].astype("string").fillna("__MISSING__")
        out[column] = pd.Categorical(
            values, categories=spec["categorical_vocabularies"][column]
        )
    return out


def apply_corrected_state_residual(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the frozen OOF-trained residual on its promoted game type only."""
    spec_path = MODEL_DIR / "corrected_state_residual_spec.json"
    model_path = MODEL_DIR / "corrected_state_residual_lgb.txt"
    if not spec_path.exists() and not model_path.exists():
        return probability
    if not spec_path.exists() or not model_path.exists():
        raise FileNotFoundError("incomplete corrected-state residual artifacts")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    features = build_corrected_state_features(frame, spec)
    model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    correction = np.asarray(
        model.predict(features, num_iteration=int(spec["num_boost_round"])),
        dtype=np.float64,
    )
    lower, upper = (float(value) for value in spec["correction_clip"])
    correction = np.clip(correction, lower, upper)
    output = np.asarray(probability, dtype=np.float64).copy()
    eligible = (
        frame["game_type"]
        .astype("string")
        .fillna("__MISSING__")
        .eq(str(spec["apply_game_type"]))
        .to_numpy()
    )
    output[eligible] = np.clip(
        output[eligible] + float(spec["eta"]) * correction[eligible],
        1e-6,
        1.0 - 1e-6,
    )
    return output


ADVANCED_RATE_COLUMNS = (
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


def _advanced_numeric(
    frame: pd.DataFrame, column: str, default: float = 0.0
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float32")
    return pd.to_numeric(frame[column], errors="coerce").astype("float32")


def _advanced_rate(
    frame: pd.DataFrame, column: str, fallback: float = 0.52
) -> pd.Series:
    return (
        _advanced_numeric(frame, column)
        .fillna(fallback)
        .clip(0.0, 1.0)
        .astype("float32")
    )


def build_advanced_domain_features(
    frame: pd.DataFrame,
    base_probability: np.ndarray,
    spec: dict,
) -> pd.DataFrame:
    """Build compact row-local domain features from frozen train endpoints."""
    corrected_spec = json.loads(
        (MODEL_DIR / "corrected_state_residual_spec.json").read_text(encoding="utf-8")
    )
    out = build_corrected_state_features(frame, corrected_spec).reset_index(drop=True)
    source = frame.reset_index(drop=True)

    for column in ("game_type", "top_bottom", "base_state", "pitcher_hand", "batter_hand"):
        out[column] = _safe_string(source[column])
    for column in ADVANCED_RATE_COLUMNS:
        values = _advanced_numeric(source, column)
        out[f"{column}_missing"] = values.isna().astype("int8")
        out[column] = values.fillna(
            0.52 if "success" in column else 0.0
        ).astype("float32")

    p_success = _advanced_rate(source, "asof_pitcher_success_rate")
    b_success = _advanced_rate(source, "asof_batter_success_rate")
    p_middle = _advanced_rate(source, "asof_pitcher_middle_rate", 0.0)
    b_middle = _advanced_rate(source, "asof_batter_middle_rate", 0.0)
    p_reverse = _advanced_rate(source, "asof_pitcher_reverse_rate", 0.0)
    p_ball = _advanced_rate(source, "asof_pitcher_ball_rate", 0.0)
    p_strike = _advanced_rate(source, "asof_pitcher_strike_rate", 0.0)
    success_windows = [
        _advanced_rate(source, f"asof_pitcher_prev{window}_game_success_rate")
        for window in (1, 3, 5)
    ]
    middle_windows = [
        _advanced_rate(
            source, f"asof_pitcher_prev{window}_game_middle_rate", 0.0
        )
        for window in (1, 3, 5)
    ]
    out["recent_success_mean"] = np.mean(success_windows, axis=0).astype("float32")
    out["recent_middle_mean"] = np.mean(middle_windows, axis=0).astype("float32")
    out["recent_success_1_minus_5"] = (success_windows[0] - success_windows[2]).astype("float32")
    out["recent_success_3_minus_5"] = (success_windows[1] - success_windows[2]).astype("float32")
    out["recent_success_3_minus_career"] = (success_windows[1] - p_success).astype("float32")
    out["recent_middle_1_minus_5"] = (middle_windows[0] - middle_windows[2]).astype("float32")
    out["recent_middle_3_minus_career"] = (middle_windows[1] - p_middle).astype("float32")
    out["pitcher_batter_success_gap"] = (p_success - b_success).astype("float32")
    out["pitcher_batter_middle_gap"] = (p_middle - b_middle).astype("float32")
    out["success_minus_strike"] = (p_success - p_strike).astype("float32")
    out["success_minus_ball"] = (p_success - p_ball).astype("float32")
    out["command_risk_sum"] = (p_reverse + p_middle + p_ball).astype("float32")
    out["command_risk_balance"] = (p_middle + p_reverse - p_strike).astype("float32")

    p_n = _advanced_numeric(source, "asof_pitcher_n").fillna(0.0).clip(lower=0.0)
    b_n = _advanced_numeric(source, "asof_batter_n").fillna(0.0).clip(lower=0.0)
    p_season_n = pd.to_numeric(out.get("pitcher_season_n", 0.0), errors="coerce").fillna(0.0)
    month = _advanced_numeric(source, "game_month", 3.0).fillna(3.0)
    elapsed_months = (month - 2.0).clip(lower=1.0)
    out["pitcher_confidence"] = (p_n / (p_n + 200.0)).astype("float32")
    out["batter_confidence"] = (b_n / (b_n + 300.0)).astype("float32")
    out["pitcher_season_confidence"] = (p_season_n / (p_season_n + 100.0)).astype("float32")
    out["season_pitch_pace"] = (p_season_n / elapsed_months).astype("float32")
    out["career_pitch_pace"] = (
        p_n / np.maximum(pd.to_numeric(source["season"], errors="coerce") - 2018, 1)
    ).astype("float32")
    out["low_history"] = p_n.lt(100).astype("int8")
    out["high_workload"] = out["season_pitch_pace"].gt(350).astype("int8")

    balls = _advanced_numeric(source, "balls_before").fillna(0.0)
    strikes = _advanced_numeric(source, "strikes_before").fillna(0.0)
    li = _advanced_numeric(source, "li").fillna(0.0).clip(lower=0.0)
    out["count_balance"] = (balls - strikes).astype("float32")
    out["pitches_in_pa_min"] = (balls + strikes).astype("float32")
    out["two_strike"] = strikes.eq(2).astype("int8")
    out["three_ball"] = balls.eq(3).astype("int8")
    out["full_count"] = (balls.eq(3) & strikes.eq(2)).astype("int8")
    out["pitcher_ahead"] = strikes.gt(balls).astype("int8")
    out["hitter_ahead"] = balls.gt(strikes).astype("int8")
    out["count_pressure"] = ((balls + strikes) * np.log1p(li)).astype("float32")
    out["three_ball_pressure"] = (balls.eq(3).astype("float32") * np.log1p(li)).astype("float32")
    out["bases_loaded"] = _advanced_numeric(source, "num_runners_on").fillna(0.0).eq(3).astype("int8")
    score_diff = _advanced_numeric(source, "score_diff_pitcher_team").fillna(0.0)
    out["absolute_score_diff"] = score_diff.abs().astype("float32")
    out["tie_game"] = score_diff.eq(0).astype("int8")
    out["win_expectancy_gap"] = (
        _advanced_numeric(source, "home_win_expectancy").fillna(50.0)
        - _advanced_numeric(source, "away_win_expectancy").fillna(50.0)
    ).astype("float32")

    p = np.clip(np.asarray(base_probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(p) != len(out):
        raise ValueError("base probability length differs from feature rows")
    out["base_probability"] = p.astype("float32")
    out["base_logit"] = np.log(p / (1.0 - p)).astype("float32")
    out["base_uncertainty"] = (p * (1.0 - p)).astype("float32")
    out["base_centered"] = (p - 0.5).astype("float32")

    missing = [column for column in spec["feature_columns"] if column not in out]
    if missing:
        raise ValueError(f"advanced-domain features missing: {missing}")
    out = out[spec["feature_columns"]].copy()
    for column in spec["categorical_columns"]:
        values = out[column].astype("string").fillna("__MISSING__")
        out[column] = pd.Categorical(
            values, categories=spec["categorical_vocabularies"][column]
        )
    return out


def apply_advanced_domain_residual(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the promoted compact OOF residual to regular-season rows."""
    spec_path = MODEL_DIR / "advanced_domain_residual_spec.json"
    model_path = MODEL_DIR / "advanced_domain_residual_lgb.txt"
    if not spec_path.exists() and not model_path.exists():
        return probability
    if not spec_path.exists() or not model_path.exists():
        raise FileNotFoundError("incomplete advanced-domain residual artifacts")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    features = build_advanced_domain_features(frame, probability, spec)
    model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    correction = np.asarray(
        model.predict(features, num_iteration=int(spec["num_boost_round"])),
        dtype=np.float64,
    )
    lower, upper = (float(value) for value in spec["correction_clip"])
    correction = np.clip(correction, lower, upper)
    output = np.asarray(probability, dtype=np.float64).copy()
    eligible = (
        frame["game_type"]
        .astype("string")
        .fillna("__MISSING__")
        .eq(str(spec["apply_game_type"]))
        .to_numpy()
    )
    output[eligible] = np.clip(
        output[eligible] + float(spec["eta"]) * correction[eligible],
        1e-6,
        1.0 - 1e-6,
    )
    return output


def build_legacy_cb_features(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Reproduce the frozen CB-R1 feature recipe with legacy season baselines."""
    corrected_spec = json.loads(
        (MODEL_DIR / "corrected_state_residual_spec.json").read_text(encoding="utf-8")
    )
    out = build_corrected_state_features(frame, corrected_spec)
    pitcher = _frozen_endpoint(frame, "pitcher_id", "legacy_cb_pitcher_prior.csv")
    batter = _frozen_endpoint(frame, "batter_id", "legacy_cb_batter_prior.csv")
    pitcher_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0)
    pitcher_count = np.rint(
        pitcher_n
        * pd.to_numeric(
            frame["asof_pitcher_success_rate"], errors="coerce"
        ).fillna(0)
    )
    batter_n = pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0)
    batter_count = np.rint(
        batter_n
        * pd.to_numeric(
            frame["asof_batter_success_rate"], errors="coerce"
        ).fillna(0)
    )
    pitcher_prior_n = pitcher["__baseline_n"].fillna(0)
    pitcher_prior_s = pitcher["__baseline_s"].fillna(0)
    batter_prior_n = batter["__baseline_n"].fillna(0)
    batter_prior_s = batter["__baseline_s"].fillna(0)
    out["pitcher_season_n"] = np.clip(
        pitcher_n.to_numpy() - pitcher_prior_n.to_numpy(), 0, None
    ).astype("float32")
    out["pitcher_season_success_count"] = np.clip(
        np.asarray(pitcher_count) - pitcher_prior_s.to_numpy(), 0, None
    ).astype("float32")
    out["batter_season_n"] = np.clip(
        batter_n.to_numpy() - batter_prior_n.to_numpy(), 0, None
    ).astype("float32")
    out["batter_season_success_count"] = np.clip(
        np.asarray(batter_count) - batter_prior_s.to_numpy(), 0, None
    ).astype("float32")
    career_rate = (np.asarray(pitcher_count) + 20.0 * 0.52) / (
        pitcher_n.to_numpy() + 20.0
    )
    season_rate = (
        out["pitcher_season_success_count"].to_numpy() + 20.0 * career_rate
    ) / (out["pitcher_season_n"].to_numpy() + 20.0)
    out["pitcher_career_rate"] = career_rate.astype("float32")
    out["pitcher_season_rate"] = season_rate.astype("float32")
    season_clipped = np.clip(season_rate, 1e-5, 1.0 - 1e-5)
    career_clipped = np.clip(career_rate, 1e-5, 1.0 - 1e-5)
    out["pitcher_season_logit_delta"] = (
        np.log(season_clipped / np.clip(1.0 - season_rate, 1e-5, 1.0))
        - np.log(career_clipped / np.clip(1.0 - career_rate, 1e-5, 1.0))
    ).astype("float32")
    out["pitcher_state_sd"] = np.sqrt(
        (season_rate * (1.0 - season_rate))
        / (out["pitcher_season_n"].to_numpy() + 20.0)
    ).astype("float32")
    out["pitcher_newcomer"] = pitcher_prior_n.eq(0).astype("int8").to_numpy()
    current_season = pd.to_numeric(frame["season"], errors="coerce").to_numpy(float)
    prior_season = pitcher["__prior_season"].to_numpy(float)
    prior_season = np.where(np.isnan(prior_season), current_season, prior_season)
    out["pitcher_long_gap"] = ((current_season - prior_season) > 1).astype("int8")

    for column in (
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "game_type",
        "top_bottom",
        "base_state",
    ):
        out[column] = _safe_string(frame[column]).to_numpy()
    missing = [column for column in spec["feature_columns"] if column not in out]
    if missing:
        raise ValueError(f"legacy CB features missing: {missing}")
    out = out[spec["feature_columns"]].copy()
    for column in spec["categorical_columns"]:
        out[column] = _safe_string(out[column])
    return out


def apply_legacy_cb_axis(
    probability: np.ndarray,
    base_probability: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Add the low-weight CB-R1 diversity effect relative to the V2 base."""
    spec_path = MODEL_DIR / "legacy_cb_axis_spec.json"
    model_path = MODEL_DIR / "legacy_cb_axis.cbm"
    if not spec_path.exists() and not model_path.exists():
        return probability
    if not spec_path.exists() or not model_path.exists():
        raise FileNotFoundError("incomplete legacy CB axis artifacts")
    from catboost import CatBoostClassifier

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    features = build_legacy_cb_features(frame, spec)
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    cb_probability = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    output = np.asarray(probability, dtype=np.float64).copy()
    baseline = np.asarray(base_probability, dtype=np.float64)
    eligible = (
        frame["game_type"]
        .astype("string")
        .fillna("__MISSING__")
        .eq(str(spec["apply_game_type"]))
        .to_numpy()
    )
    output[eligible] = np.clip(
        output[eligible]
        + float(spec["effect_weight"])
        * (cb_probability[eligible] - baseline[eligible]),
        1e-6,
        1.0 - 1e-6,
    )
    return output


def resolve_trackman_weights(frame: pd.DataFrame, hybrid: dict) -> np.ndarray:
    """Resolve a scalar or row-local game-type Trackman blend weight.

    The optional mapping is deliberately keyed only by ``game_type``.  It does
    not inspect test frequencies, ordering, or any other batch-level statistic.
    Existing packages without the mapping retain their scalar behavior.
    """
    default = float(hybrid.get("trackman_default_weight", hybrid.get("trackman_weight", 0.0)))
    mapping = hybrid.get("trackman_weight_by_game_type")
    if mapping is None:
        weights = np.full(len(frame), default, dtype=np.float64)
    else:
        if not isinstance(mapping, dict):
            raise ValueError("trackman_weight_by_game_type must be a mapping")
        if not np.isfinite(default) or not 0.0 <= default <= 1.0:
            raise ValueError("trackman default weight must be in [0, 1]")
        weights = np.full(len(frame), default, dtype=np.float64)
        game_type = frame["game_type"].astype("string").fillna("__MISSING__")
        for value, raw_weight in mapping.items():
            weight = float(raw_weight)
            if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("trackman game-type weights must be in [0, 1]")
            mask = game_type.eq(str(value)).to_numpy()
            weights[mask] = weight
    return weights


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    feature_spec = json.loads((MODEL_DIR / "feature_spec.json").read_text(encoding="utf-8"))
    ensemble = json.loads((MODEL_DIR / "ensemble.json").read_text(encoding="utf-8"))
    features = build_features(frame, feature_spec)
    # model_str avoids Unicode-path limitations in the LightGBM Windows C API.
    booster = lgb.Booster(model_str=(MODEL_DIR / "lgb_model.txt").read_text(encoding="utf-8"))
    lgb_probability = booster.predict(
        features, num_iteration=int(ensemble["num_iterations"])
    )
    calibrated = apply_calibration(
        np.asarray(lgb_probability, dtype=float),
        ensemble["calibration"],
        float(ensemble["global_rate"]),
    )
    weights = ensemble["weights"]
    prediction = float(weights["lightgbm"]) * calibrated
    rf_probability = None
    if float(weights["hierarchical_prior"]) > 0.0:
        prediction += float(weights["hierarchical_prior"]) * hierarchical_prior(
            frame,
            float(ensemble["global_rate"]),
            float(ensemble["prior_alpha"]),
            float(ensemble["prior_batter_weight"]),
        )
    if float(weights["random_forest"]) > 0.0:
        import joblib

        rf = joblib.load(MODEL_DIR / "rf_model.joblib")
        rf_probability = rf.predict_proba(frame[ensemble["official_features"]])[:, 1]
        rf_probability = apply_calibration(
            np.asarray(rf_probability, dtype=float),
            ensemble.get("rf_calibration", {"method": "identity"}),
            float(ensemble["global_rate"]),
        )
        prediction += float(weights["random_forest"]) * rf_probability
    hybrid_path = MODEL_DIR / "hybrid.json"
    if hybrid_path.exists():
        if rf_probability is None:
            raise ValueError("hybrid candidate requires incumbent random forest")
        import joblib

        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        candidate_prediction = np.asarray(prediction, dtype=np.float64).copy()
        regular = frame["game_type"].astype("string").fillna("__MISSING__").eq(
            str(
                hybrid.get(
                    "r_only_apply_game_type",
                    hybrid.get("recency_apply_game_type", "R"),
                )
            )
        ).to_numpy()
        if hybrid.get("r_only_rf_model"):
            r_only_rf = joblib.load(MODEL_DIR / hybrid["r_only_rf_model"])
            r_only_probability = r_only_rf.predict_proba(
                frame[ensemble["official_features"]]
            )[:, 1]
            r_only_probability = apply_calibration(
                np.asarray(r_only_probability, dtype=float),
                hybrid.get("r_only_rf_calibration", {"method": "identity"}),
                float(ensemble["global_rate"]),
            )
            r_only_weight = float(hybrid.get("r_only_weight", 0.0))
            candidate_prediction[regular] = (
                (1.0 - r_only_weight) * candidate_prediction[regular]
                + r_only_weight * r_only_probability[regular]
            )
        elif hybrid.get("recency_rf_model"):
            weighted_rf = joblib.load(MODEL_DIR / hybrid["recency_rf_model"])
            weighted_rf_probability = weighted_rf.predict_proba(
                frame[ensemble["official_features"]]
            )[:, 1]
            weighted_rf_probability = apply_calibration(
                np.asarray(weighted_rf_probability, dtype=float),
                ensemble.get("rf_calibration", {"method": "identity"}),
                float(ensemble["global_rate"]),
            )
            candidate_prediction[regular] += float(weights["random_forest"]) * (
                weighted_rf_probability[regular] - rf_probability[regular]
            )

        trackman_weights = resolve_trackman_weights(frame, hybrid)
        if np.any(trackman_weights > 0.0):
            trackman_spec = json.loads(
                (MODEL_DIR / hybrid["trackman_feature_spec"]).read_text(
                    encoding="utf-8"
                )
            )
            trackman_features = build_features(frame, trackman_spec)
            trackman_booster = lgb.Booster(
                model_str=(MODEL_DIR / hybrid["trackman_lgb_model"]).read_text(
                    encoding="utf-8"
                )
            )
            trackman_raw = trackman_booster.predict(
                trackman_features,
                num_iteration=int(hybrid["trackman_num_iterations"]),
            )
            drift_predictions = [
                apply_calibration(
                    np.asarray(trackman_raw, dtype=float),
                    {"method": "logit_offset", "offset": float(offset)},
                    float(trackman_spec["global_rate"]),
                )
                for offset in hybrid["trackman_drift_offsets"]
            ]
            trackman_probability = np.average(
                np.column_stack(drift_predictions),
                axis=1,
                weights=np.asarray(hybrid["trackman_drift_weights"], dtype=float),
            )
            prediction = (
                (1.0 - trackman_weights) * candidate_prediction
                + trackman_weights * trackman_probability
            )
        else:
            prediction = candidate_prediction
    prediction = apply_optional_game_type_offsets(
        np.asarray(prediction, dtype=np.float64), frame
    )
    base_prediction = np.asarray(prediction, dtype=np.float64).copy()
    if (MODEL_DIR / "advanced_domain_residual_spec.json").exists():
        prediction = apply_advanced_domain_residual(base_prediction, frame)
    else:
        prediction = apply_corrected_state_residual(base_prediction, frame)
    prediction = apply_legacy_cb_axis(prediction, base_prediction, frame)
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains NaN or infinite values")
    return np.clip(prediction, 0.0, 1.0)


def main() -> None:
    started = time.perf_counter()
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    if ID_COL not in test.columns:
        raise ValueError(f"test is missing {ID_COL}")
    if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
        raise ValueError("test row_id must be non-null and unique")
    row_ids = test[ID_COL].copy()
    prediction = predict_dataframe(test)
    if len(prediction) != len(test):
        raise ValueError("prediction length differs from test")
    submission = pd.DataFrame({ID_COL: row_ids.to_numpy(copy=True), TARGET_COL: prediction})
    if submission[ID_COL].tolist() != row_ids.tolist():
        raise ValueError("row order changed during inference")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved {OUTPUT_PATH} rows={len(submission)} "
        f"elapsed={time.perf_counter() - started:.3f}s"
    )


if __name__ == "__main__":
    main()
