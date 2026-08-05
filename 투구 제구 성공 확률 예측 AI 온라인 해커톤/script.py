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
    if feature_set in {"engineered", "trackman"}:
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
    prediction = np.asarray(prediction, dtype=np.float64)
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
