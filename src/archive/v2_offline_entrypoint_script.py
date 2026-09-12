"""Offline evaluation entry point for the early v2 package.

Archived from the repository root. Paths resolve against the project root,
so it reads ./data/test.csv and writes ./output/submission.csv there.
Every feature is row-local or uses training artifacts inside ./model; no
test-set aggregate, frequency, order, or distribution is used.
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
BASE_DIR = Path(__file__).resolve().parents[2]
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
    """Mirror src.archive.features.FeatureBuilder using only frozen train artifacts."""
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
            mapping = {
                str(key): int(value) for key, value in category_maps[column].items()
            }
            out[column] = (
                _safe_string(out[column]).map(mapping).fillna(-1).astype("int32")
            )
        else:
            out[column] = _safe_string(out[column]).astype(str)
    for column in out.columns:
        if column not in spec["categorical_columns"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out


def resolve_trackman_weights(frame: pd.DataFrame, hybrid: dict) -> np.ndarray:
    """Resolve frozen row-local blend weights, optionally by game type."""
    default = float(
        hybrid.get("trackman_default_weight", hybrid.get("trackman_weight", 0.0))
    )
    weights = np.full(len(frame), default, dtype=np.float64)
    mapping = hybrid.get("trackman_weight_by_game_type")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("trackman_weight_by_game_type must be a mapping")
        game_type = frame["game_type"].astype("string").fillna("__MISSING__")
        for value, raw_weight in mapping.items():
            weights[game_type.eq(str(value)).to_numpy()] = float(raw_weight)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Trackman weights must be finite and non-negative")
    return weights


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
