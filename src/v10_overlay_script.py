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


def build_features(
    frame: pd.DataFrame,
    spec: dict,
    *,
    encode_categories: bool = True,
) -> pd.DataFrame:
    """Mirror src.features.FeatureBuilder using only frozen train artifacts."""
    base_columns = list(spec["base_columns"])
    missing = [column for column in base_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"test columns missing: {missing}")
    out = frame[base_columns].copy()
    drop_columns = [column for column in spec.get("drop_columns", []) if column in out]
    if drop_columns:
        out = out.drop(columns=drop_columns)
    feature_set = spec["feature_set"]
    if feature_set == "no_asof":
        out = out.drop(columns=[column for column in out if column.startswith("asof_")])
    if feature_set in {
        "engineered",
        "trackman",
        "trackman_pitcher",
        "hierarchical_v2",
    }:
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
    if feature_set == "hierarchical_v2":
        count = _safe_string(out["balls_before"]) + "-" + _safe_string(
            out["strikes_before"]
        )
        out["pitcher_game_type"] = _safe_string(out["pitcher_id"]) + "-" + _safe_string(
            out["game_type"]
        )
        out["pitcher_team_game_type"] = _safe_string(
            out["pitcher_team_id"]
        ) + "-" + _safe_string(out["game_type"])
        out["batter_pitcher_hand"] = _safe_string(out["batter_id"]) + "-" + _safe_string(
            out["pitcher_hand"]
        )
        out["count_base_state"] = count + "-" + _safe_string(out["base_state"])
        runner_state = _safe_string(out["base_state"])
        out["inning_runner_state"] = _safe_string(out["inning"]) + "-" + runner_state
        leverage_bucket = (
            pd.cut(
                pd.to_numeric(out["li"], errors="coerce"),
                bins=[-np.inf, 0.75, 1.5, 3.0, np.inf],
                labels=["low", "medium", "high", "very_high"],
            )
            .astype("string")
            .fillna("__MISSING__")
        )
        out["leverage_runner_state"] = leverage_bucket + "-" + runner_state

        rate = float(spec["global_rate"])
        pn = (
            pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        bn = (
            pd.to_numeric(out["asof_batter_n"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        pmn = (
            pd.to_numeric(out["asof_pitcher_pitchmix_n"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        pr = (
            pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
            .fillna(rate)
            .clip(1e-5, 1.0 - 1e-5)
        )
        br = (
            pd.to_numeric(out["asof_batter_success_rate"], errors="coerce")
            .fillna(rate)
            .clip(1e-5, 1.0 - 1e-5)
        )
        global_logit = float(np.log(rate / (1.0 - rate)))
        for prefix, count_values, success_rate in (
            ("pitcher", pn, pr),
            ("batter", bn, br),
        ):
            alpha = 200.0
            posterior = (count_values * success_rate + alpha * rate) / (
                count_values + alpha
            )
            out[f"{prefix}_posterior_rate"] = posterior.astype("float32")
            out[f"{prefix}_posterior_se"] = np.sqrt(
                posterior * (1.0 - posterior) / (count_values + alpha + 1.0)
            ).astype("float32")
            posterior_clip = posterior.clip(1e-5, 1.0 - 1e-5)
            out[f"{prefix}_posterior_logit_residual"] = (
                np.log(posterior_clip / (1.0 - posterior_clip)) - global_logit
            ).astype("float32")
            out[f"{prefix}_history_confidence"] = (
                count_values / (count_values + alpha)
            ).astype("float32")
        recent = out[
            [
                "asof_pitcher_prev1_game_success_rate",
                "asof_pitcher_prev3_game_success_rate",
                "asof_pitcher_prev5_game_success_rate",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        out["pitcher_recent_stability"] = (
            recent.std(axis=1, skipna=True).fillna(0.0).astype("float32")
        )
        out["pitcher_recent_available"] = recent.notna().sum(axis=1).astype("int8")
        out["pitchmix_history_confidence"] = (pmn / (pmn + 100.0)).astype("float32")
        out["pitcher_cold_start"] = (pn <= 0).astype("int8")
        out["batter_cold_start"] = (bn <= 0).astype("int8")
        out["entity_backoff_level"] = np.select(
            [pn >= 200, pn >= 30, bn >= 30],
            ["pitcher_reliable", "pitcher_sparse", "batter_only"],
            default="global",
        )
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
        if encode_categories:
            mapping = {str(key): int(value) for key, value in category_maps[column].items()}
            out[column] = _safe_string(out[column]).map(mapping).fillna(-1).astype("int32")
        else:
            out[column] = _safe_string(out[column]).astype(str)
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


RECENT_NUMERIC_COLUMNS = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
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
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
RECENT_PITCHER_COMPONENTS = [
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
RECENT_STABLE_STATS = ("rate", "rel", "delta", "post_sd", "log_n", "delta_rel")


def _recent_numeric(
    frame: pd.DataFrame, column: str, default: float = np.nan
) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _recent_rounded_count(n: np.ndarray, rate: np.ndarray) -> np.ndarray:
    output = np.zeros(len(n), dtype=np.float64)
    valid = (n > 0) & np.isfinite(rate)
    output[valid] = np.rint(n[valid] * rate[valid])
    return output


def _recent_current_state(
    rows: pd.DataFrame, bank: dict[str, object]
) -> pd.DataFrame:
    rows = rows.reset_index(drop=True)
    output = pd.DataFrame(index=np.arange(len(rows)))
    prior = float(bank["prior"])

    def add_entity(
        entity: str,
        n_column: str,
        rate_column: str,
        prefix: str,
        reliability_strength: float,
    ) -> None:
        ids = rows[entity]
        history_n = ids.map(bank[f"{prefix}_n"]).fillna(0.0).to_numpy(np.float64)
        history_s = ids.map(bank[f"{prefix}_s"]).fillna(0.0).to_numpy(np.float64)
        cumulative_n = np.maximum(
            np.nan_to_num(_recent_numeric(rows, n_column, 0.0)), 0.0
        )
        cumulative_rate = _recent_numeric(rows, rate_column, prior)
        cumulative_rate = np.where(np.isfinite(cumulative_rate), cumulative_rate, prior)
        cumulative_s = _recent_rounded_count(cumulative_n, cumulative_rate)
        season_n = np.maximum(cumulative_n - history_n, 0.0)
        season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
        raw = np.divide(
            season_s,
            season_n,
            out=np.full(len(rows), prior, dtype=np.float64),
            where=season_n > 0,
        )
        for strength in (40.0, 80.0, 160.0):
            output[f"season__{prefix}_rate_k{int(strength)}"] = (
                season_s + strength * prior
            ) / (season_n + strength)
        output[f"season__{prefix}_raw"] = raw
        output[f"season__{prefix}_log_n"] = np.log1p(season_n)
        output[f"season__{prefix}_reliability"] = season_n / (
            season_n + reliability_strength
        )
        output[f"season__{prefix}_minus_career"] = raw - cumulative_rate

    add_entity(
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "pitcher",
        80.0,
    )
    add_entity(
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "batter",
        100.0,
    )
    output["season__pitcher_minus_batter_k80"] = (
        output["season__pitcher_rate_k80"] - output["season__batter_rate_k80"]
    )
    ids = rows["pitcher_id"]
    cumulative_n = np.maximum(
        np.nan_to_num(_recent_numeric(rows, "asof_pitcher_n", 0.0)), 0.0
    )
    base_n = ids.map(bank["component_base_n"]).fillna(0.0).to_numpy(np.float64)
    window_n = np.maximum(cumulative_n - base_n, 0.0)
    components = {}
    for column in RECENT_PITCHER_COMPONENTS:
        component_prior = float(bank["component_prior"].get(column, 0.5))
        rate = _recent_numeric(rows, column, component_prior)
        rate = np.where(np.isfinite(rate), rate, component_prior)
        cumulative = _recent_rounded_count(cumulative_n, rate)
        baseline = (
            ids.map(bank["component_base"].get(column, {}))
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        window_count = np.clip(cumulative - baseline, 0.0, window_n)
        components[column] = (window_count + 60.0 * component_prior) / (
            window_n + 60.0
        )
    output["season__component_log_n"] = np.log1p(window_n)
    output["season__component_reliability"] = window_n / (window_n + 60.0)
    output["season__strike_rate"] = components["asof_pitcher_strike_rate"]
    output["season__ball_rate"] = components["asof_pitcher_ball_rate"]
    output["season__strike_ball_margin"] = (
        output["season__strike_rate"] - output["season__ball_rate"]
    )
    output["season__middle_rate"] = components["asof_pitcher_middle_rate"]
    output["season__reverse_rate"] = components["asof_pitcher_reverse_rate"]
    mix = np.column_stack(
        [
            components["asof_pitcher_fastball_rate"],
            components["asof_pitcher_breaking_rate"],
            components["asof_pitcher_offspeed_rate"],
        ]
    )
    mix = np.clip(mix, 1e-6, 1.0)
    mix /= mix.sum(axis=1, keepdims=True)
    output["season__fastball_rate"] = mix[:, 0]
    output["season__breaking_rate"] = mix[:, 1]
    output["season__offspeed_rate"] = mix[:, 2]
    output["season__pitchmix_entropy"] = -np.sum(mix * np.log(mix), axis=1)
    return output.astype(np.float32)


def _recent_row_state(rows: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(rows)))
    pitcher_n = np.maximum(
        np.nan_to_num(_recent_numeric(rows, "asof_pitcher_n", 0.0)), 0.0
    )
    batter_n = np.maximum(
        np.nan_to_num(_recent_numeric(rows, "asof_batter_n", 0.0)), 0.0
    )
    career = _recent_numeric(rows, "asof_pitcher_success_rate", 0.5)
    prev1 = _recent_numeric(rows, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _recent_numeric(rows, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _recent_numeric(rows, "asof_pitcher_prev5_game_success_rate", 0.5)
    recent = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5
    output["self__pitcher_log_n"] = np.log1p(pitcher_n)
    output["self__batter_log_n"] = np.log1p(batter_n)
    output["self__pitcher_reliability"] = pitcher_n / (pitcher_n + 160.0)
    output["self__batter_reliability"] = batter_n / (batter_n + 160.0)
    output["self__recent_blend"] = recent
    output["self__recent_minus_career"] = recent - career
    output["self__prev1_minus_prev5"] = prev1 - prev5
    output["self__strike_ball_margin"] = _recent_numeric(
        rows, "asof_pitcher_strike_rate", 0.33
    ) - _recent_numeric(rows, "asof_pitcher_ball_rate", 0.33)
    output["self__count_pressure"] = (
        (rows["balls_before"].to_numpy() == 3)
        | (rows["strikes_before"].to_numpy() == 2)
    ).astype(np.float32)
    output["self__same_hand"] = (
        rows["pitcher_hand"].astype(str).to_numpy()
        == rows["batter_hand"].astype(str).to_numpy()
    ).astype(np.float32)
    return output.astype(np.float32)


def _recent_exact_matrix(
    frame: pd.DataFrame,
    preprocess: dict,
    *,
    include_categories: bool,
) -> pd.DataFrame:
    numeric = [column for column in RECENT_NUMERIC_COLUMNS if column in frame]
    output = pd.concat(
        [
            frame[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True),
            _recent_row_state(frame),
            _recent_current_state(frame, preprocess["exact_bank"]),
        ],
        axis=1,
    )
    if include_categories:
        categories = pd.DataFrame(index=np.arange(len(frame)))
        for column, mapping in preprocess["category_maps"].items():
            if column == "domain3" and column not in frame:
                regular = frame["game_type"].eq("R")
                anchor = frame["pitcher_team_id"].eq(13) | frame[
                    "batter_team_id"
                ].eq(13)
                values = pd.Series(
                    np.where(
                        ~regular.to_numpy(),
                        "F",
                        np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE"),
                    ),
                    index=frame.index,
                    dtype="string",
                )
            else:
                values = frame[column]
            categories[f"cat__{column}"] = (
                values
                .astype("string")
                .fillna("<NA>")
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(np.float32)
                .to_numpy()
            )
        output = pd.concat([output, categories], axis=1)
    expected = (
        preprocess["exact_lgb_columns"]
        if include_categories
        else preprocess["exact_ridge_columns"]
    )
    if list(output.columns) != list(expected):
        raise ValueError("recent exact-ASOF feature columns differ")
    return output.astype(np.float32)


def _recent_stable_lookup(
    rows: pd.DataFrame,
    table: pd.DataFrame,
    keys: list[str],
    prefix: str,
    global_rate: float,
) -> pd.DataFrame:
    merged = rows[keys].reset_index(drop=True).merge(
        table, on=keys, how="left", sort=False, validate="many_to_one"
    )
    output = pd.DataFrame(index=np.arange(len(rows)))
    for name in RECENT_STABLE_STATS:
        fill = global_rate if name == "rate" else 0.0
        output[f"{prefix}__{name}"] = merged[name].fillna(fill).to_numpy(np.float64)
    return output


def _recent_stable_matrix(
    rows: pd.DataFrame, base_probability: np.ndarray, preprocess: dict
) -> pd.DataFrame:
    stable_rows = rows.reset_index(drop=True).copy()
    stable_rows["pressure"] = np.where(
        stable_rows["balls_before"].to_numpy() == 3,
        "threeball",
        np.where(
            stable_rows["strikes_before"].to_numpy() == 2,
            "twostrike",
            "normal",
        ),
    )
    bank = preprocess["stable_bank"]
    global_rate = float(bank["global_rate"])
    blocks = [
        _recent_stable_lookup(
            stable_rows,
            bank["pitcher"],
            ["pitcher_id"],
            "pitcher",
            global_rate,
        ),
        _recent_stable_lookup(
            stable_rows,
            bank["pitcher_hand"],
            ["pitcher_id", "batter_hand"],
            "pitcher_hand",
            global_rate,
        ),
        _recent_stable_lookup(
            stable_rows,
            bank["pitcher_pressure"],
            ["pitcher_id", "pressure"],
            "pitcher_pressure",
            global_rate,
        ),
        _recent_stable_lookup(
            stable_rows,
            bank["pitcher_pressure_hand"],
            ["pitcher_id", "pressure", "batter_hand"],
            "pitcher_pressure_hand",
            global_rate,
        ),
    ]
    output = pd.concat(blocks, axis=1)
    output["base_probability"] = np.asarray(base_probability, dtype=np.float64)
    if list(output.columns) != list(preprocess["stable_columns"]):
        raise ValueError("recent stable feature columns differ")
    return output


def _v14_trend_matrix(frame: pd.DataFrame, preprocess: dict) -> pd.DataFrame:
    numeric = [column for column in RECENT_NUMERIC_COLUMNS if column in frame]
    output = frame[numeric].apply(pd.to_numeric, errors="coerce").reset_index(
        drop=True
    )
    prev1 = _recent_numeric(frame, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _recent_numeric(frame, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _recent_numeric(frame, "asof_pitcher_prev5_game_success_rate", 0.5)
    recent = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5
    output["eng__recent"] = recent
    output["eng__recent_delta"] = recent - _recent_numeric(
        frame, "asof_pitcher_success_rate", 0.5
    )
    output["eng__strike_ball"] = _recent_numeric(
        frame, "asof_pitcher_strike_rate", 0.33
    ) - _recent_numeric(frame, "asof_pitcher_ball_rate", 0.33)
    output["eng__same_hand"] = (
        frame["pitcher_hand"].astype(str).to_numpy()
        == frame["batter_hand"].astype(str).to_numpy()
    ).astype(np.float32)
    output["eng__close"] = (
        np.abs(_recent_numeric(frame, "score_diff_pitcher_team", 0.0)) <= 1
    ).astype(np.float32)
    output["eng__late"] = (
        _recent_numeric(frame, "inning", 1.0) >= 7
    ).astype(np.float32)

    categories = pd.DataFrame(index=np.arange(len(frame)))
    regular = frame["game_type"].eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    for column, mapping in preprocess["trend_category_maps"].items():
        if column == "domain3" and column not in frame:
            values = pd.Series(
                np.where(
                    ~regular.to_numpy(),
                    "F",
                    np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE"),
                ),
                index=frame.index,
                dtype="string",
            )
        else:
            values = frame[column]
        categories[f"cat__{column}"] = (
            values
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
            .to_numpy()
        )
    output = pd.concat([output, categories], axis=1).astype(np.float32)
    if list(output.columns) != list(preprocess["trend_columns"]):
        raise ValueError("v14 trend feature columns differ")
    return output


def apply_v14_refinement(
    probability: np.ndarray,
    frame: pd.DataFrame,
    exact_lgb: np.ndarray,
    exact_numeric: pd.DataFrame,
) -> np.ndarray:
    spec_path = MODEL_DIR / "v14_refinement_spec.json"
    if not spec_path.exists():
        return probability
    import joblib

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    preprocess = joblib.load(MODEL_DIR / "v14_refinement_preprocess.joblib")
    output = np.asarray(probability, dtype=np.float64).copy()
    base = output.copy()
    regular = frame["game_type"].eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(int(spec["anchor_team_id"]))
        | frame["batter_team_id"].eq(int(spec["anchor_team_id"]))
    ).to_numpy()
    pitcher_reliability = exact_numeric[
        "season__pitcher_reliability"
    ].to_numpy(np.float64)
    batter_reliability = exact_numeric[
        "season__batter_reliability"
    ].to_numpy(np.float64)
    threshold = float(spec["anchor_reliability_threshold"])
    anchor_gate = (
        anchor
        & (pitcher_reliability >= threshold)
        & (batter_reliability >= threshold)
    )
    if anchor_gate.any():
        expected = list(preprocess["anchor_columns"])
        if list(exact_numeric.columns) != expected:
            raise ValueError("v14 anchor feature columns differ")
        anchor_raw = exact_numeric.fillna(preprocess["anchor_median"]).to_numpy(
            np.float64
        )
        anchor_x = (
            anchor_raw - np.asarray(preprocess["anchor_mean"], dtype=np.float64)
        ) / np.asarray(preprocess["anchor_scale"], dtype=np.float64)
        anchor_model = joblib.load(MODEL_DIR / "v14_anchor_ridge.joblib")
        anchor_probability = np.clip(anchor_model.predict(anchor_x), 0.001, 0.999)
        weight = float(spec["anchor_weight"])
        output[anchor_gate] = np.clip(
            base[anchor_gate]
            + weight * (anchor_probability[anchor_gate] - base[anchor_gate]),
            1e-6,
            1.0 - 1e-6,
        )

    finals = ~regular
    if finals.any() and float(spec["f_trend_alpha"]) != 0.0:
        trend_features = _v14_trend_matrix(frame, preprocess)
        trend_booster = lgb.Booster(
            model_str=(MODEL_DIR / "v14_f_trend_lgb.txt").read_text(
                encoding="utf-8"
            )
        )
        residual = np.asarray(
            trend_booster.predict(
                trend_features,
                num_iteration=int(spec["f_trend_num_iterations"]),
            ),
            dtype=np.float64,
        )
        game_type = frame["game_type"].astype("string").fillna("__MISSING__")
        domain = np.where(
            game_type.ne("R").to_numpy(),
            "F",
            np.where(anchor, "R_ANCHOR", "R_CORE"),
        )
        source_means = {
            str(key): float(value)
            for key, value in preprocess["trend_source_means"].items()
        }
        audit_prior = np.asarray(
            [source_means.get(str(value), 0.5) for value in domain],
            dtype=np.float64,
        ) + float(preprocess["trend_slope"])
        trend_probability = np.clip(audit_prior + residual, 0.001, 0.999)
        adjustment = (
            float(spec["parent_f_exact_weight"])
            * float(spec["f_trend_alpha"])
            * (trend_probability - np.asarray(exact_lgb, dtype=np.float64))
        )
        output[finals] = np.clip(
            output[finals] + adjustment[finals], 1e-6, 1.0 - 1e-6
        )
    return output


def apply_v16_residual(probability: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Apply a frozen previous-season empirical-Bayes correction to R_CORE."""
    spec_path = MODEL_DIR / "v16_residual_spec.json"
    if not spec_path.exists():
        return probability
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("method") != "smoothed_oof_residual_mean":
        raise ValueError("unsupported v16 residual method")
    group_columns = list(spec["group_columns"])
    supported = (
        ["pitcher_id", "batter_hand"],
        ["pitcher_id", "batter_hand", "pressure"],
    )
    if group_columns not in supported:
        raise ValueError("unsupported v16 residual group columns")
    separator = "\x1f"
    key_parts = [
        frame["pitcher_id"].astype("string").fillna("__MISSING__"),
        frame["batter_hand"].astype("string").fillna("__MISSING__"),
    ]
    if "pressure" in group_columns:
        balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy()
        strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy()
        pressure = pd.Series(
            np.where(
                balls == 3,
                "threeball",
                np.where(strikes == 2, "twostrike", "normal"),
            ),
            index=frame.index,
            dtype="string",
        )
        key_parts.append(pressure)
    keys = key_parts[0]
    for part in key_parts[1:]:
        keys = keys + separator + part
    effects = pd.Series(spec["effects"], dtype="float64")
    correction = keys.map(effects).fillna(float(spec.get("unseen_effect", 0.0))).to_numpy(
        np.float64
    )
    game_type = frame["game_type"].astype("string").fillna("__MISSING__")
    anchor_team = int(spec["anchor_team_id"])
    anchor = (
        frame["pitcher_team_id"].eq(anchor_team)
        | frame["batter_team_id"].eq(anchor_team)
    ).to_numpy()
    core = game_type.eq("R").to_numpy() & ~anchor
    output = np.asarray(probability, dtype=np.float64).copy()
    output[core] = np.clip(
        output[core] + correction[core], 1e-6, 1.0 - 1e-6
    )
    return output


def apply_recent_exact_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    spec_path = MODEL_DIR / "recent_exact_spec.json"
    if not spec_path.exists():
        return probability
    import joblib

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    preprocess = joblib.load(MODEL_DIR / "recent_exact_preprocess.joblib")
    lgb_features = _recent_exact_matrix(frame, preprocess, include_categories=True)
    booster = lgb.Booster(
        model_str=(MODEL_DIR / "recent_exact_lgb.txt").read_text(encoding="utf-8")
    )
    exact_lgb = np.asarray(
        booster.predict(
            lgb_features, num_iteration=int(spec["exact_lgb_num_iterations"])
        ),
        dtype=np.float64,
    )
    ridge_features = _recent_exact_matrix(frame, preprocess, include_categories=False)
    ridge_raw = ridge_features.fillna(preprocess["exact_ridge_median"]).to_numpy(
        np.float64
    )
    ridge_x = np.clip(
        (ridge_raw - np.asarray(preprocess["exact_ridge_mean"], dtype=np.float64))
        / np.asarray(preprocess["exact_ridge_scale"], dtype=np.float64),
        -8.0,
        8.0,
    )
    exact_ridge_model = joblib.load(MODEL_DIR / "recent_exact_ridge.joblib")
    exact_ridge = np.clip(exact_ridge_model.predict(ridge_x), 0.001, 0.999)

    output = np.asarray(probability, dtype=np.float64).copy()
    base = output.copy()
    game_type = frame["game_type"].astype("string").fillna("__MISSING__")
    anchor = (
        frame["pitcher_team_id"].eq(int(spec["anchor_team_id"]))
        | frame["batter_team_id"].eq(int(spec["anchor_team_id"]))
    ).to_numpy()
    core = game_type.eq("R").to_numpy() & ~anchor
    finals = game_type.eq("F").to_numpy()
    if core.any():
        core_probability = (
            base[core]
            + float(spec["core_bias"])
            + float(spec["core_exact_lgb_weight"])
            * (exact_lgb[core] - base[core])
            + float(spec["core_exact_ridge_weight"])
            * (exact_ridge[core] - base[core])
        )
        stable_features = _recent_stable_matrix(
            frame.loc[core].reset_index(drop=True), base[core], preprocess
        )
        stable_x = (
            stable_features.to_numpy(np.float64)
            - np.asarray(preprocess["stable_mean"], dtype=np.float64)
        ) / np.asarray(preprocess["stable_scale"], dtype=np.float64)
        stable_model = joblib.load(MODEL_DIR / "recent_stable_ridge.joblib")
        stable_conditional = stable_model.predict(stable_x) - float(
            stable_model.intercept_
        )
        core_probability += float(spec["core_stable_gamma"]) * stable_conditional
        output[core] = np.clip(core_probability, 1e-6, 1.0 - 1e-6)
    if finals.any():
        weight = float(spec["f_exact_lgb_weight"])
        output[finals] = np.clip(
            base[finals] + weight * (exact_lgb[finals] - base[finals]),
            1e-6,
            1.0 - 1e-6,
        )
    refined = apply_v14_refinement(output, frame, exact_lgb, ridge_features)
    return apply_v16_residual(refined, frame)


def _joint_domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    return np.where(~regular.to_numpy(), "F", np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE"))


def _joint_features(frame: pd.DataFrame, preprocess: dict) -> tuple[pd.DataFrame, np.ndarray]:
    rows = frame.reset_index(drop=True).copy()
    domain = _joint_domain(rows)
    rows["domain3"] = domain
    numeric = [column for column in RECENT_NUMERIC_COLUMNS if column in rows]
    output = pd.concat(
        [
            rows[numeric].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            _recent_current_state(rows, preprocess["bank"]),
            _recent_row_state(rows),
        ],
        axis=1,
    )
    categories = pd.DataFrame(index=np.arange(len(rows)))
    for feature_name in preprocess["categorical_columns"]:
        column = feature_name.removeprefix("cat__")
        values = rows[column].astype("string").fillna("__MISSING__")
        categories[feature_name] = pd.Categorical(
            values.astype(str), categories=preprocess["categories"][feature_name]
        )
    output = pd.concat([output, categories], axis=1)
    if list(output.columns) != list(preprocess["feature_columns"]):
        raise ValueError("joint state/mode feature columns differ")
    return output, domain


def _joint_booster(filename: str) -> lgb.Booster:
    return lgb.Booster(model_str=(MODEL_DIR / filename).read_text(encoding="utf-8"))


def _joint_power(probability: np.ndarray, power: float) -> np.ndarray:
    output = np.power(np.clip(probability, 1e-8, 1.0), power)
    return output / output.sum(axis=1, keepdims=True)


def _joint_config_state_raw(
    raw: dict[str, np.ndarray], raw_name: str
) -> np.ndarray:
    prefix = "pair_selected__"
    if raw_name.startswith(prefix):
        other = raw_name[len(prefix) :]
        return 0.5 * (raw["selected_h05_l15_b075"] + raw[other])
    return raw[raw_name]


def apply_joint_state_mode_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the frozen row-local 2025 joint state/failure-mode route."""
    spec_path = MODEL_DIR / "joint_state_mode_spec.json"
    preprocess_path = MODEL_DIR / "joint_state_mode_preprocess.joblib"
    if not spec_path.exists() and not preprocess_path.exists():
        return probability
    if not spec_path.exists() or not preprocess_path.exists():
        raise FileNotFoundError("incomplete joint state/mode artifacts")
    import joblib

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    v20_spec_path = MODEL_DIR / "v20_target1160_spec.json"
    v20_spec = (
        json.loads(v20_spec_path.read_text(encoding="utf-8"))
        if v20_spec_path.exists()
        else None
    )
    preprocess = joblib.load(preprocess_path)
    features, domain = _joint_features(frame, preprocess)
    incumbent = np.asarray(probability, dtype=np.float64)
    pitcher = features["season__pitcher_rate_k80"].to_numpy(np.float64)
    batter = features["season__batter_rate_k80"].to_numpy(np.float64)

    state_raw: dict[str, np.ndarray] = {}
    for name in spec["state_order"]:
        model_spec = spec["state_models"][name]
        pitcher_weight = float(model_spec["pitcher_weight"])
        baseline = pitcher_weight * pitcher + (1.0 - pitcher_weight) * batter
        if "file" in model_spec:
            residual = _joint_booster(model_spec["file"]).predict(features)
            state_raw[name] = np.clip(baseline + residual, 0.001, 0.999)
        else:
            value = baseline.copy()
            for domain_name, filename in model_spec["files"].items():
                mask = domain == domain_name
                if not mask.any():
                    continue
                value[mask] = np.clip(
                    baseline[mask]
                    + _joint_booster(filename).predict(features.loc[mask]),
                    0.001,
                    0.999,
                )
            state_raw[name] = value
    individual_correction = np.column_stack(
        [state_raw[name] - incumbent for name in spec["state_order"]]
    )

    mode_probability = {}
    for name, filename in spec["mode_classifiers"].items():
        mode_probability[name] = np.asarray(
            _joint_booster(filename).predict(features), dtype=np.float64
        )
    mean_mode_probability = np.mean(
        np.stack(list(mode_probability.values())), axis=0
    )
    outcome = _joint_booster(spec["mode_outcome"])
    outcome_baseline = 0.75 * pitcher + 0.25 * batter
    raw_by_mode = []
    for mode_name in spec["mode_names"]:
        typed = features.copy()
        typed["cat__latent_failure_mode"] = pd.Categorical(
            np.full(len(typed), mode_name, dtype=object),
            categories=preprocess["mode_categories"],
        )
        raw_by_mode.append(
            np.clip(outcome_baseline + outcome.predict(typed), 0.001, 0.999)
        )
    raw_by_mode = np.column_stack(raw_by_mode)
    conditional_success = np.asarray(spec["conditional_success"], dtype=np.float64)

    mode_signal_names = [str(choice["mode_name"]) for choice in spec["choices"]]
    if v20_spec is not None:
        mode_signal_names.append(str(v20_spec["extra_mode"]["name"]))
    mode_raw = {}
    for name in mode_signal_names:
        if name in mode_raw:
            continue
        conditional = name.startswith("conditional_")
        core_name = name.removeprefix("conditional_")
        if "_pow" not in core_name:
            raise ValueError(f"unsupported joint mode signal: {name}")
        probability_name, power_text = core_name.rsplit("_pow", 1)
        power = float(power_text)
        if probability_name == "mode_lgb_mean":
            selected_probability = mean_mode_probability
        elif probability_name.startswith("mode_lgb_h"):
            selected_probability = mode_probability[
                probability_name.removeprefix("mode_lgb_")
            ]
        else:
            raise ValueError(f"unsupported joint mode probability: {probability_name}")
        selected_probability = _joint_power(selected_probability, power)
        mode_raw[name] = (
            selected_probability @ conditional_success
            if conditional
            else np.sum(selected_probability * raw_by_mode, axis=1)
        )

    abs_ranges = {
        "allmag": (0.0, np.inf),
        "lt005": (0.0, 0.005),
        "005to020": (0.005, 0.020),
        "020to040": (0.020, 0.040),
        "ge040": (0.040, np.inf),
        "lt020": (0.0, 0.020),
        "lt040": (0.0, 0.040),
    }
    bands = {
        "p_all": (0.0, 1.0),
        "p_20_80": (0.2, 0.8),
        "p_30_70": (0.3, 0.7),
        "p_40_60": (0.4, 0.6),
    }
    output = incumbent.copy()
    for choice in spec["choices"]:
        raw_name, sign, abs_name, band_name, agreement_text, config_weight = str(
            choice["state_config"]
        ).split("|")
        selected_raw = _joint_config_state_raw(state_raw, raw_name)
        correction = selected_raw - incumbent
        magnitude = np.abs(correction)
        agreement = np.where(
            (correction >= 0.0)[:, None],
            individual_correction >= 0.0,
            individual_correction < 0.0,
        ).mean(axis=1)
        apply_mask = domain == str(choice["domain"])
        if sign == "positive":
            apply_mask &= correction > 0.0
        elif sign == "negative":
            apply_mask &= correction < 0.0
        low_abs, high_abs = abs_ranges[abs_name]
        low_p, high_p = bands[band_name]
        apply_mask &= (magnitude >= low_abs) & (magnitude < high_abs)
        apply_mask &= (incumbent >= low_p) & (incumbent <= high_p)
        apply_mask &= agreement >= float(agreement_text) - 1e-12
        state_delta = (
            float(choice["state_multiplier"])
            * float(config_weight)
            * correction
        )
        domain_mask = domain == str(choice["domain"])
        state_delta[~apply_mask] = 0.0
        output[domain_mask] = np.clip(
            incumbent[domain_mask]
            + state_delta[domain_mask]
            + float(choice["mode_multiplier"])
            * (mode_raw[str(choice["mode_name"])][domain_mask] - incumbent[domain_mask]),
            0.001,
            0.999,
        )
    if v20_spec is not None:
        extra_mode = v20_spec["extra_mode"]
        extra_name = str(extra_mode["name"])
        output = np.clip(
            output
            + float(extra_mode["weight"]) * (mode_raw[extra_name] - incumbent),
            0.001,
            0.999,
        )
    return output


def apply_v20_target1160_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply frozen 2024 OOF EB and training-only TrackMan distillation."""
    spec_path = MODEL_DIR / "v20_target1160_spec.json"
    if not spec_path.exists():
        return probability
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    separator = str(spec["separator"])
    domain = _joint_domain(frame)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy()
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy()
    pressure = pd.Series(
        np.where(
            balls == 3,
            "threeball",
            np.where(strikes == 2, "twostrike", "normal"),
        ),
        index=frame.index,
        dtype="string",
    )
    key_frame = frame.copy()
    key_frame["pressure"] = pressure
    correction = np.zeros(len(frame), dtype=np.float64)
    for recipe in spec["eb_recipes"]:
        columns = [str(value) for value in recipe["columns"]]
        pieces = [
            key_frame[column].astype("string").fillna("__MISSING__")
            for column in columns
        ]
        keys = pieces[0]
        for piece in pieces[1:]:
            keys = keys + separator + piece
        effects = pd.Series(recipe["effects"], dtype="float64")
        values = keys.map(effects).fillna(0.0).to_numpy(np.float64)
        apply_mask = np.ones(len(frame), dtype=bool)
        if str(recipe["domain"]) != "ALL":
            apply_mask &= domain == str(recipe["domain"])
        correction[apply_mask] += float(recipe["weight"]) * values[apply_mask]

    pfd = spec["pfd"]
    columns = [str(value) for value in pfd["feature_columns"]]
    features = frame.loc[:, columns].copy()
    categorical = {str(value) for value in pfd["categorical_columns"]}
    for column in columns:
        if column in categorical:
            values = features[column].astype("string").fillna("__MISSING__")
            features[column] = pd.Categorical(
                values.astype(str), categories=pfd["categories"][column]
            )
        else:
            features[column] = pd.to_numeric(
                features[column], errors="coerce"
            ).astype(np.float32)
    control = np.asarray(
        _joint_booster(str(pfd["control_model"])).predict(
            features, num_iteration=int(pfd["num_iterations"]["control"])
        ),
        dtype=np.float64,
    )
    soft = np.asarray(
        _joint_booster(str(pfd["soft_model"])).predict(
            features, num_iteration=int(pfd["num_iterations"]["soft"])
        ),
        dtype=np.float64,
    )
    correction += float(pfd["overlay_weight"]) * (soft - control)
    return np.clip(
        np.asarray(probability, dtype=np.float64) + correction,
        0.001,
        0.999,
    )


def apply_v21_context_state_eb_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply frozen train-only recency EB maps using current-row features."""
    spec_path = MODEL_DIR / "v21_context_state_eb_spec.json"
    if not spec_path.exists():
        return probability
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    separator = str(spec["separator"])
    key_frame = frame.copy()

    def rate_bin(column: str, bins: int) -> np.ndarray:
        value = pd.to_numeric(key_frame[column], errors="coerce").fillna(0.5)
        return np.floor(
            np.clip(value.to_numpy(np.float64), 0.0, 1.0) * bins
        ).astype(np.int16)

    key_frame["prev3_b10"] = rate_bin(
        "asof_pitcher_prev3_game_success_rate", 10
    )
    key_frame["prev5_b10"] = rate_bin(
        "asof_pitcher_prev5_game_success_rate", 10
    )
    key_frame["prev3_b20"] = rate_bin(
        "asof_pitcher_prev3_game_success_rate", 20
    )
    key_frame["reverse_b10"] = rate_bin("asof_pitcher_reverse_rate", 10)
    key_frame["inning_band"] = pd.cut(
        pd.to_numeric(key_frame["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, 9, np.inf),
        labels=("early", "middle", "late", "extra"),
    ).astype("string")
    domain = _joint_domain(frame)
    correction = np.zeros(len(frame), dtype=np.float64)
    for recipe in spec["recipes"]:
        columns = [str(value) for value in recipe["columns"]]
        pieces = [
            key_frame[column].astype("string").fillna("__MISSING__")
            for column in columns
        ]
        keys = pieces[0]
        for piece in pieces[1:]:
            keys = keys + separator + piece
        effect = pd.Series(recipe["effects"], dtype="float64")
        value = keys.map(effect).fillna(0.0).to_numpy(np.float64)
        apply_mask = np.ones(len(frame), dtype=bool)
        if str(recipe["domain"]) != "ALL":
            apply_mask &= domain == str(recipe["domain"])
        correction[apply_mask] += float(recipe["weight"]) * value[apply_mask]
    return np.clip(
        np.asarray(probability, dtype=np.float64) + correction, 0.001, 0.999
    )


def apply_v22_low_variance_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the frozen row-local domain calibration and ASOF prior blend."""
    spec_path = MODEL_DIR / "v22_low_variance_spec.json"
    if not spec_path.exists():
        return probability
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    parent = np.asarray(probability, dtype=np.float64)
    domain = _joint_domain(frame)
    correction = np.zeros(len(frame), dtype=np.float64)
    for name, parameters in spec["domain_calibration"].items():
        mask = domain == str(name)
        correction[mask] += float(parameters["weight"]) * (
            float(parameters["anchor"]) - parent[mask]
        )
    pitcher = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(float(spec["missing_rate_default"]))
        .to_numpy(np.float64)
    )
    batter = (
        pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce")
        .fillna(float(spec["missing_rate_default"]))
        .to_numpy(np.float64)
    )
    prior = (
        float(spec["asof_prior"]["pitcher_fraction"]) * pitcher
        + float(spec["asof_prior"]["batter_fraction"]) * batter
    )
    correction += float(spec["asof_prior"]["weight"]) * (prior - parent)
    return np.clip(parent + correction, 0.001, 0.999)


def _v25_postbreak_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror the frozen post-break training features using row-local inputs."""
    output = frame.copy()
    output["domain3"] = _joint_domain(frame).astype(str)
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string")
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        output[f"log1p_{column}"] = np.log1p(
            pd.to_numeric(output[column], errors="coerce").clip(lower=0.0)
        )
    output["recent_success_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    output["recent_middle_delta_1_5"] = (
        pd.to_numeric(output["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
        - pd.to_numeric(output["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    )
    output["pitcher_batter_rate_gap"] = (
        pd.to_numeric(output["asof_pitcher_success_rate"], errors="coerce")
        - pd.to_numeric(output["asof_batter_success_rate"], errors="coerce")
    )
    return output


def apply_v25_postbreak_anchor_overlay(
    probability: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    """Blend a 2024-only direct model on R_ANCHOR rows."""
    spec_path = MODEL_DIR / "v25_postbreak_anchor_spec.json"
    if not spec_path.exists():
        return probability
    import joblib

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    parent = np.asarray(probability, dtype=np.float64)
    domain = _joint_domain(frame)
    apply_mask = domain == str(spec["apply_domain"])
    if not apply_mask.any():
        return parent
    model = joblib.load(MODEL_DIR / str(spec["model_file"]))
    selected = frame.loc[apply_mask].copy()
    features = _v25_postbreak_frame(selected)
    direct = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    low, high = (float(value) for value in spec["probability_clip"])
    direct = np.clip(direct, low, high)
    result = parent.copy()
    eta = float(spec["blend_eta"])
    result[apply_mask] = np.clip(
        parent[apply_mask] + eta * (direct - parent[apply_mask]), low, high
    )
    return result


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

        catboost_models = hybrid.get("f_catboost_models", [])
        if catboost_models:
            from catboost import CatBoostClassifier

            catboost_spec = json.loads(
                (MODEL_DIR / hybrid["f_catboost_feature_spec"]).read_text(
                    encoding="utf-8"
                )
            )
            game_type = frame["game_type"].astype("string").fillna("__MISSING__")
            f_mask = game_type.eq(str(hybrid.get("f_catboost_game_type", "F"))).to_numpy()
            if f_mask.any():
                catboost_features = build_features(
                    frame.loc[f_mask].reset_index(drop=True),
                    catboost_spec,
                    encode_categories=False,
                )
                model_predictions = []
                for model_name in catboost_models:
                    catboost_model = CatBoostClassifier()
                    catboost_model.load_model(str(MODEL_DIR / str(model_name)))
                    model_predictions.append(
                        catboost_model.predict_proba(catboost_features)[:, 1]
                    )
                catboost_probability = np.average(
                    np.column_stack(model_predictions),
                    axis=1,
                    weights=np.asarray(
                        hybrid.get(
                            "f_catboost_model_weights",
                            np.ones(len(model_predictions), dtype=float),
                        ),
                        dtype=float,
                    ),
                )
                catboost_probability = apply_calibration(
                    np.asarray(catboost_probability, dtype=float),
                    hybrid.get("f_catboost_calibration", {"method": "identity"}),
                    float(catboost_spec["global_rate"]),
                )
                catboost_weight = float(hybrid.get("f_catboost_weight", 1.0))
                prediction[f_mask] = (
                    (1.0 - catboost_weight) * prediction[f_mask]
                    + catboost_weight * catboost_probability
                )
    prediction = apply_optional_game_type_offsets(
        np.asarray(prediction, dtype=np.float64), frame
    )
    base_prediction = np.asarray(prediction, dtype=np.float64).copy()
    if (MODEL_DIR / "advanced_domain_residual_spec.json").exists():
        prediction = apply_advanced_domain_residual(base_prediction, frame)
    else:
        prediction = apply_corrected_state_residual(base_prediction, frame)
    prediction = apply_legacy_cb_axis(prediction, base_prediction, frame)
    prediction = apply_recent_exact_overlay(prediction, frame)
    prediction = apply_joint_state_mode_overlay(prediction, frame)
    prediction = apply_v20_target1160_overlay(prediction, frame)
    prediction = apply_v21_context_state_eb_overlay(prediction, frame)
    prediction = apply_v22_low_variance_overlay(prediction, frame)
    prediction = apply_v25_postbreak_anchor_overlay(prediction, frame)
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
