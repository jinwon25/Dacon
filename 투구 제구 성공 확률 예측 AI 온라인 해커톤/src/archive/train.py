"""Run comparable baselines, calibration, blending, and final training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.archive.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    choose_base_rate_shrinkage,
    choose_three_way_blend,
)
from src.archive.data import ID_COL, TARGET_COL, read_main, read_trackman
from src.archive.experiment_tracking import append_experiment
from src.archive.features import (
    TRACKMAN_MEASURES,
    FeatureBuilder,
    build_trackman_context,
    empirical_pitcher_prior,
    hierarchical_prior,
)
from src.metrics import brier_score, brier_skill_score
from src.archive.validation import season_holdout

SEED = 42
PRIMARY_SPLIT = "season_forward_train_2019_2023_valid_2024"
CALIBRATION_SOURCE = "season_forward_train_2019_2022_valid_2023"
RF_CAT_COLS = ["top_bottom", "game_type", "base_state"]


class MemoryMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, process.memory_info().rss)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024.0**2)


def _brier_eval(prediction: np.ndarray, dataset: lgb.Dataset):
    return "brier", float(np.mean((prediction - dataset.get_label()) ** 2)), False


def _base_lgb_params(variant: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    seed = int(variant.get("seed", seed))
    return {
        "objective": variant.get("objective", "binary"),
        "metric": "None",
        "verbosity": -1,
        "num_threads": 6,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_leaves": int(variant["num_leaves"]),
        "learning_rate": float(variant["learning_rate"]),
        "min_data_in_leaf": int(variant["min_data_in_leaf"]),
        "lambda_l2": float(variant["lambda_l2"]),
        "feature_fraction": float(variant["feature_fraction"]),
        "bagging_fraction": float(variant["bagging_fraction"]),
        "bagging_freq": 1,
        "max_bin": 255,
    }


def train_lgb_holdout(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    variant: dict[str, Any],
    max_rounds: int,
    early_stopping_rounds: int,
    trackman_context: pd.DataFrame | None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    feature_set = variant["feature_set"]
    builder = FeatureBuilder(
        feature_set=feature_set,
        trackman_context=trackman_context,
        drop_columns=list(variant.get("drop_columns", [])),
    )
    y_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder.fit(train.iloc[train_idx], y_train)
    with MemoryMonitor() as monitor:
        build_start = time.perf_counter()
        x_train = builder.transform(train.iloc[train_idx])
        x_valid = builder.transform(train.iloc[valid_idx])
        build_seconds = time.perf_counter() - build_start
        categorical = [col for col in builder.categorical_columns if col in x_train.columns]
        dtrain = lgb.Dataset(
            x_train,
            label=y_train,
            weight=sample_weight,
            categorical_feature=categorical,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            x_valid,
            label=y_valid,
            categorical_feature=categorical,
            reference=dtrain,
            free_raw_data=True,
        )
        fit_start = time.perf_counter()
        booster = lgb.train(
            _base_lgb_params(variant),
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            valid_names=["valid"],
            feval=_brier_eval,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        prediction = booster.predict(x_valid, num_iteration=booster.best_iteration)
        if variant.get("objective") == "regression":
            prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        inference_seconds = time.perf_counter() - inference_start
        model_size_mb = len(booster.model_to_string(num_iteration=booster.best_iteration).encode("utf-8")) / (1024.0**2)
        feature_names = x_train.columns.tolist()
        del x_train, x_valid, dtrain, dvalid
        gc.collect()
    return {
        "builder": builder,
        "booster": booster,
        "prediction": prediction,
        "target": y_valid,
        "brier": brier_score(y_valid, prediction),
        "skill": brier_skill_score(y_valid, prediction),
        "best_iteration": int(booster.best_iteration),
        "build_seconds": build_seconds,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": monitor.peak_mb,
        "feature_names": feature_names,
    }


def make_official_rf(features: list[str], random_state: int = SEED) -> Pipeline:
    numeric = [col for col in features if col not in RF_CAT_COLS]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                RF_CAT_COLS,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )
    return Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=200,
                    n_jobs=6,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_rf_holdout(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    features: list[str],
    random_state: int = SEED,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    model = make_official_rf(features, random_state=random_state)
    with MemoryMonitor() as monitor:
        start = time.perf_counter()
        fit_params = (
            {"clf__sample_weight": sample_weight}
            if sample_weight is not None
            else {}
        )
        model.fit(
            train.iloc[train_idx][features],
            train.iloc[train_idx][TARGET_COL],
            **fit_params,
        )
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict_proba(train.iloc[valid_idx][features])[:, 1]
        inference_seconds = time.perf_counter() - start
    target = train.iloc[valid_idx][TARGET_COL].to_numpy()
    serialized = joblib.dump(model, os.devnull) if os.name != "nt" else None
    # Measuring with an in-memory pickle avoids retaining another on-disk candidate.
    import pickle

    model_size_mb = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / (1024.0**2)
    return {
        "model": model,
        "prediction": prediction,
        "target": target,
        "brier": brier_score(target, prediction),
        "skill": brier_skill_score(target, prediction),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": monitor.peak_mb,
    }


def log_result(
    experiments_path: Path,
    experiment_id: str,
    feature_groups: str,
    model_params: dict[str, Any] | str,
    raw_brier: float,
    skill: float,
    *,
    calibrated_brier: float | str = "",
    calibration: str = "none",
    train_seconds: float = 0.0,
    inference_seconds: float = 0.0,
    model_size_mb: float = 0.0,
    peak_memory_mb: float = 0.0,
    leakage_notes: str = "row-local official features only",
    split: str = PRIMARY_SPLIT,
) -> None:
    append_experiment(
        experiments_path,
        {
            "experiment_id": experiment_id,
            "validation_split": split,
            "feature_groups": feature_groups,
            "model_params": json.dumps(model_params, ensure_ascii=False, sort_keys=True)
            if isinstance(model_params, dict)
            else model_params,
            "seed": SEED,
            "raw_brier_score": f"{raw_brier:.9f}",
            "calibrated_brier_score": f"{calibrated_brier:.9f}"
            if isinstance(calibrated_brier, float)
            else calibrated_brier,
            "local_brier_skill_score": f"{skill:.3f}",
            "calibration": calibration,
            "training_seconds": f"{train_seconds:.3f}",
            "inference_seconds": f"{inference_seconds:.3f}",
            "model_size_mb": f"{model_size_mb:.3f}",
            "peak_memory_mb": f"{peak_memory_mb:.1f}",
            "leakage_risk_notes": leakage_notes,
        },
    )


def _metric_row(name: str, prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "brier": brier_score(target, prediction),
        "skill": brier_skill_score(target, prediction),
    }


def _build_trackman_lookup(project_dir: Path) -> pd.DataFrame:
    path = project_dir / "model" / "trackman_context.csv"
    usecols = [
        "season",
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitch_type_group",
        *TRACKMAN_MEASURES,
    ]
    print("Building prior-season Trackman context...")
    trackman = read_trackman(project_dir / "data" / "trackman_history.csv", usecols=usecols)
    context = build_trackman_context(trackman)
    context.to_csv(path, index=False, encoding="utf-8")
    del trackman
    gc.collect()
    print(f"Trackman context: {len(context)} rows -> {path}")
    return context


def _render_findings(
    path: Path,
    results: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    best_name: str,
    best_iteration: int,
    final_choice: str,
    trackman_promoted: bool,
) -> None:
    def table(rows: list[dict[str, Any]]) -> str:
        lines = ["| model | Brier Score | local Brier Skill Score |", "| --- | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['model']} | {row['brier']:.9f} | {row['skill']:.3f} |")
        return "\n".join(lines)

    content = f"""# 초기 분석 및 실험 결과

모든 아래 성능은 동일한 primary validation인 2019~2023 학습 / 2024 검증에서 실제 실행한 값이다. 서로 다른 split의 점수를 직접 비교하지 않았다.

## 검증 선택

- 비공개 평가는 2025년이고 학습의 최신 시즌은 2024년이므로 1년 forward holdout을 선택했다.
- 메인 데이터에 경기 ID와 정확한 날짜가 없어 game-group split은 구성할 수 없다. season 경계가 같은 경기 혼입을 차단한다.
- calibration과 blend 파라미터는 2019~2022 학습 모델의 2023 OOF 예측에서 학습하고 2024에 고정 적용했다.
- 모든 범주 사전과 base rate는 fold train에서만 학습했다. test 행 간 집계는 없다.

## Primary 모델 비교

{table(results)}

## Calibration 및 blend 비교

{table(calibration_rows)}

## 선택

- 최선 LightGBM variant: `{best_name}` (2024 early-stop iteration {best_iteration}).
- 최종 추론 구조: **{final_choice}**.
- Trackman context 승격 여부: **{trackman_promoted}**. 선수 ID 직접 조인은 0%라 금지했으며, context도 이전 시즌 league 집계만 사용했다.

## 한계와 누수 위험

- 2024 단일 시즌 holdout이라 시즌별 모델 순위 안정성을 완전히 확인하지 못했다.
- `asof_*`는 공식적으로 허용된 투구 직전 피처이고 갱신식도 확인했지만 target 관련 과거 집계라 분포 변화에 민감하다. no-asof ablation을 비교 기준으로 유지한다.
- 실제 2025 test의 선수 cold-start 비율은 5행 형식 샘플로 추정할 수 없다.
- Trackman 선수/팀 매핑은 제공되지 않아 선수 단위 물리 피처를 만들지 않았다.
"""
    path.write_text(content, encoding="utf-8")


def run(project_dir: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments_path = project_dir / "research" / "reports" / "experiments.csv"
    model_dir = project_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading optimized train data...")
    train = read_main(project_dir / "data" / "train.csv")
    features = [col for col in read_main(project_dir / "data" / "test.csv", nrows=0).columns if col != ID_COL]
    train_idx, valid_idx = season_holdout(train, int(config["primary_validation_season"]))
    y_train = train.iloc[train_idx][TARGET_COL].to_numpy()
    y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy()
    global_rate = float(y_train.mean())
    results: list[dict[str, Any]] = []

    # Baseline 0: global, empirical pitcher, and Bayesian hierarchical priors.
    prior_candidates: dict[str, np.ndarray] = {
        "constant_train_rate": np.full(len(valid_idx), global_rate),
        "empirical_pitcher_prior": empirical_pitcher_prior(train.iloc[valid_idx], global_rate),
    }
    for alpha in (50.0, 200.0, 1000.0):
        prior_candidates[f"hierarchical_prior_a{int(alpha)}"] = hierarchical_prior(
            train.iloc[valid_idx], global_rate, alpha=alpha
        )
    for name, prediction in prior_candidates.items():
        row = _metric_row(name, prediction, y_valid)
        results.append(row)
        log_result(
            experiments_path,
            name,
            "official asof row-local prior" if name != "constant_train_rate" else "none",
            {"global_rate": global_rate},
            row["brier"],
            row["skill"],
            leakage_notes="official shifted asof rates; no validation target used" if name != "constant_train_rate" else "none",
        )
        print(name, row)
    best_prior_name = min(
        (name for name in prior_candidates if name.startswith("hierarchical")),
        key=lambda name: brier_score(y_valid, prior_candidates[name]),
    )
    best_prior_alpha = float(best_prior_name.rsplit("a", 1)[1])
    prior_2024 = prior_candidates[best_prior_name]

    # Baseline 1: exact official RandomForest recipe.
    print("Training official RandomForest on primary split...")
    rf_2024 = train_rf_holdout(train, train_idx, valid_idx, features)
    rf_row = {
        "model": "official_random_forest",
        "brier": rf_2024["brier"],
        "skill": rf_2024["skill"],
    }
    results.append(rf_row)
    log_result(
        experiments_path,
        "official_random_forest",
        "47 official features; 3 string categoricals",
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 200},
        rf_row["brier"],
        rf_row["skill"],
        train_seconds=rf_2024["fit_seconds"],
        inference_seconds=rf_2024["inference_seconds"],
        model_size_mb=rf_2024["model_size_mb"],
        peak_memory_mb=rf_2024["peak_memory_mb"],
        leakage_notes="official reproduction; numeric player IDs are treated as continuous as in baseline",
    )
    print("official_random_forest", rf_row)

    context = _build_trackman_lookup(project_dir)

    # Baseline 2: purposeful LightGBM variants.
    lgb_results: dict[str, dict[str, Any]] = {}
    for variant in config["lightgbm_variants"]:
        print(f"Training {variant['name']} ({variant['feature_set']})...")
        result = train_lgb_holdout(
            train,
            train_idx,
            valid_idx,
            variant,
            int(config["max_boost_rounds"]),
            int(config["early_stopping_rounds"]),
            context if variant["feature_set"] == "trackman" else None,
        )
        lgb_results[variant["name"]] = result
        row = {"model": variant["name"], "brier": result["brier"], "skill": result["skill"]}
        results.append(row)
        log_result(
            experiments_path,
            variant["name"],
            variant["feature_set"],
            {**variant, "best_iteration": result["best_iteration"]},
            result["brier"],
            result["skill"],
            train_seconds=result["fit_seconds"] + result["build_seconds"],
            inference_seconds=result["inference_seconds"],
            model_size_mb=result["model_size_mb"],
            peak_memory_mb=result["peak_memory_mb"],
            leakage_notes=(
                "Trackman uses only seasons before each row season; no player join"
                if variant["feature_set"] == "trackman"
                else "category maps and base rate fit on fold train only; row-local transform"
            ),
        )
        print(variant["name"], row, "iteration", result["best_iteration"])

    best_name = min(lgb_results, key=lambda name: lgb_results[name]["brier"])
    best_2024 = lgb_results[best_name]
    best_variant = next(item for item in config["lightgbm_variants"] if item["name"] == best_name)
    print("Best LightGBM:", best_name, best_2024["brier"])

    # Train the selected configurations one season earlier. These 2023 OOF
    # predictions are the only data used to choose calibration/blend parameters.
    cal_train_idx, cal_valid_idx = season_holdout(train, int(config["calibration_source_season"]))
    print("Training best LightGBM for 2023 calibration OOF...")
    best_2023 = train_lgb_holdout(
        train,
        cal_train_idx,
        cal_valid_idx,
        best_variant,
        int(config["max_boost_rounds"]),
        int(config["early_stopping_rounds"]),
        context if best_variant["feature_set"] == "trackman" else None,
    )
    print("Training official RF for 2023 blend OOF...")
    rf_2023 = train_rf_holdout(train, cal_train_idx, cal_valid_idx, features)
    y_2023 = train.iloc[cal_valid_idx][TARGET_COL].to_numpy()
    global_2023 = float(train.iloc[cal_train_idx][TARGET_COL].mean())
    prior_2023 = hierarchical_prior(
        train.iloc[cal_valid_idx], global_2023, alpha=best_prior_alpha
    )

    platt = PlattCalibrator().fit(best_2023["prediction"], y_2023)
    isotonic = IsotonicCalibrator().fit(best_2023["prediction"], y_2023)
    cal_predictions_2023 = {
        "raw": best_2023["prediction"],
        "platt": platt.predict(best_2023["prediction"]),
        "isotonic": isotonic.predict(best_2023["prediction"]),
    }
    cal_predictions_2024 = {
        "raw": best_2024["prediction"],
        "platt": platt.predict(best_2024["prediction"]),
        "isotonic": isotonic.predict(best_2024["prediction"]),
    }
    shrink_weight, _ = choose_base_rate_shrinkage(
        best_2023["prediction"], y_2023, global_2023
    )
    cal_predictions_2024["base_rate_shrink"] = (
        (1.0 - shrink_weight) * best_2024["prediction"] + shrink_weight * global_rate
    )
    calibration_rows = [
        _metric_row(f"best_lgb_{method}", prediction, y_valid)
        for method, prediction in cal_predictions_2024.items()
    ]
    best_cal_method = min(calibration_rows, key=lambda row: row["brier"])["model"].replace("best_lgb_", "")
    if best_cal_method == "base_rate_shrink":
        selected_2023 = (
            (1.0 - shrink_weight) * best_2023["prediction"] + shrink_weight * global_2023
        )
        selected_2024 = cal_predictions_2024["base_rate_shrink"]
    else:
        selected_2023 = cal_predictions_2023[best_cal_method]
        selected_2024 = cal_predictions_2024[best_cal_method]

    blend_weights, _ = choose_three_way_blend(
        [selected_2023, rf_2023["prediction"], prior_2023], y_2023, step=0.05
    )
    blend_2024 = (
        blend_weights[0] * selected_2024
        + blend_weights[1] * rf_2024["prediction"]
        + blend_weights[2] * prior_2024
    )
    calibration_rows.append(_metric_row("fixed_2023_oof_three_way_blend", blend_2024, y_valid))
    for row in calibration_rows:
        method = row["model"].replace("best_lgb_", "")
        log_result(
            experiments_path,
            row["model"],
            f"{best_name} calibration/blend",
            {"source": "2023_oof", "blend_weights": blend_weights, "shrink_weight": shrink_weight},
            best_2024["brier"],
            row["skill"],
            calibrated_brier=row["brier"],
            calibration=method,
            leakage_notes="parameters fit only on 2023 OOF and applied unchanged to 2024",
        )

    calibrated_best_row = min(calibration_rows[:-1], key=lambda row: row["brier"])
    blend_row = calibration_rows[-1]
    use_blend = blend_row["brier"] < calibrated_best_row["brier"]
    final_choice = blend_row["model"] if use_blend else calibrated_best_row["model"]

    # Refit the selected calibrator method on primary 2024 OOF for the final
    # 2025 inference model. The method itself was selected using untouched 2024.
    if best_cal_method == "platt":
        final_calibrator = PlattCalibrator().fit(best_2024["prediction"], y_valid)
        calibration_spec = final_calibrator.to_dict()
    elif best_cal_method == "isotonic":
        final_calibrator = IsotonicCalibrator().fit(best_2024["prediction"], y_valid)
        calibration_spec = final_calibrator.to_dict()
    elif best_cal_method == "base_rate_shrink":
        final_calibrator = None
        calibration_spec = {"method": "base_rate_shrink", "weight": shrink_weight}
    else:
        final_calibrator = None
        calibration_spec = {"method": "identity"}

    final_rounds = max(1, int(round((best_2023["best_iteration"] + best_2024["best_iteration"]) / 2)))
    print(f"Training final {best_name} for {final_rounds} rounds...")
    final_builder = FeatureBuilder(
        feature_set=best_variant["feature_set"],
        trackman_context=context if best_variant["feature_set"] == "trackman" else None,
    )
    y_all = train[TARGET_COL].to_numpy(dtype=np.int8)
    x_all = final_builder.fit_transform(train, y_all)
    final_dataset = lgb.Dataset(
        x_all,
        label=y_all,
        categorical_feature=final_builder.categorical_columns,
        free_raw_data=True,
    )
    final_booster = lgb.train(
        _base_lgb_params(best_variant),
        final_dataset,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(100)],
    )
    # LightGBM's Windows C API cannot write paths containing Korean text.
    # Python handles the same path correctly, so serialize to a string first.
    (model_dir / "lgb_model.txt").write_text(
        final_booster.model_to_string(), encoding="utf-8"
    )
    final_builder.save_spec(model_dir / "feature_spec.json")
    del x_all, final_dataset
    gc.collect()

    include_rf = bool(use_blend and blend_weights[1] > 0.0)
    if include_rf:
        print("Training final RandomForest blend member...")
        final_rf = make_official_rf(features)
        final_rf.fit(train[features], train[TARGET_COL])
        joblib.dump(final_rf, model_dir / "rf_model.joblib", compress=3)

    ensemble_spec = {
        "calibration": calibration_spec,
        "weights": {
            "lightgbm": blend_weights[0] if use_blend else 1.0,
            "random_forest": blend_weights[1] if use_blend else 0.0,
            "hierarchical_prior": blend_weights[2] if use_blend else 0.0,
        },
        "prior_alpha": best_prior_alpha,
        "prior_batter_weight": 0.25,
        "global_rate": float(train[TARGET_COL].mean()),
        "feature_set": best_variant["feature_set"],
        "best_variant": best_name,
        "num_iterations": final_rounds,
        "official_features": features,
    }
    (model_dir / "ensemble.json").write_text(
        json.dumps(ensemble_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "primary_validation": PRIMARY_SPLIT,
        "best_lgb": best_name,
        "best_lgb_brier_2024": best_2024["brier"],
        "calibration_method": best_cal_method,
        "blend_weights_selected_on_2023": blend_weights,
        "final_choice": final_choice,
        "trackman_promoted": best_variant["feature_set"] == "trackman",
        "library_versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _render_findings(
        project_dir / "research" / "reports" / "initial_findings.md",
        results,
        calibration_rows,
        best_name,
        best_2024["best_iteration"],
        final_choice,
        best_variant["feature_set"] == "trackman",
    )
    print("Primary results:")
    for row in results:
        print(row)
    print("Calibration/blend:")
    for row in calibration_rows:
        print(row)
    print("Final model artifacts written to", model_dir)


def finalize_from_logged_results(
    project_dir: Path,
    config_path: Path,
    num_rounds: int | None = None,
) -> None:
    """Recover final artifacts without repeating already logged experiments."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments = pd.read_csv(project_dir / "research" / "reports" / "experiments.csv")
    lgb_rows = experiments[
        experiments["experiment_id"].isin(item["name"] for item in config["lightgbm_variants"])
    ].copy()
    if lgb_rows.empty:
        raise RuntimeError("no completed LightGBM experiment exists")
    lgb_rows = lgb_rows.drop_duplicates("experiment_id", keep="last")
    best_row = lgb_rows.loc[lgb_rows["raw_brier_score"].astype(float).idxmin()]
    best_name = str(best_row["experiment_id"])
    best_variant = next(item for item in config["lightgbm_variants"] if item["name"] == best_name)
    params_logged = json.loads(best_row["model_params"])
    rounds = int(num_rounds or params_logged["best_iteration"])

    calibration = experiments[experiments["experiment_id"].str.startswith("best_lgb_", na=False)].copy()
    calibration = calibration.drop_duplicates("experiment_id", keep="last")
    calibration["effective_brier"] = pd.to_numeric(calibration["calibrated_brier_score"], errors="coerce")
    best_cal = calibration.loc[calibration["effective_brier"].idxmin()]
    cal_method = str(best_cal["calibration"])
    # The completed run selected raw/identity. Non-identity methods require OOF
    # calibrator coefficients, so fail loudly rather than silently refit in this
    # recovery path.
    if cal_method != "raw":
        raise RuntimeError(
            f"final-only recovery currently requires raw calibration, got {cal_method}; rerun full training"
        )

    print(f"Final-only: {best_name}, rounds={rounds}, calibration=identity")
    train = read_main(project_dir / "data" / "train.csv")
    test_columns = read_main(project_dir / "data" / "test.csv", nrows=0).columns.tolist()
    features = [col for col in test_columns if col != ID_COL]
    model_dir = project_dir / "model"
    context = None
    if best_variant["feature_set"] == "trackman":
        context = pd.read_csv(model_dir / "trackman_context.csv")
    builder = FeatureBuilder(feature_set=best_variant["feature_set"], trackman_context=context)
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    with MemoryMonitor() as monitor:
        x_all = builder.fit_transform(train, target)
        dataset = lgb.Dataset(
            x_all,
            label=target,
            categorical_feature=builder.categorical_columns,
            free_raw_data=True,
        )
        booster = lgb.train(
            _base_lgb_params(best_variant),
            dataset,
            num_boost_round=rounds,
            callbacks=[lgb.log_evaluation(100)],
        )
    (model_dir / "lgb_model.txt").write_text(booster.model_to_string(), encoding="utf-8")
    builder.save_spec(model_dir / "feature_spec.json")
    ensemble_spec = {
        "calibration": {"method": "identity"},
        "weights": {"lightgbm": 1.0, "random_forest": 0.0, "hierarchical_prior": 0.0},
        "prior_alpha": 50.0,
        "prior_batter_weight": 0.25,
        "global_rate": float(target.mean()),
        "feature_set": best_variant["feature_set"],
        "best_variant": best_name,
        "num_iterations": rounds,
        "official_features": features,
    }
    (model_dir / "ensemble.json").write_text(
        json.dumps(ensemble_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "primary_validation": PRIMARY_SPLIT,
        "best_lgb": best_name,
        "best_lgb_brier_2024": float(best_row["raw_brier_score"]),
        "calibration_method": "identity",
        "final_choice": "best_lgb_raw",
        "trackman_promoted": best_variant["feature_set"] == "trackman",
        "num_iterations": rounds,
        "final_training_peak_memory_mb": monitor.peak_mb,
        "library_versions": {
            "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__
        },
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary_ids = [
        "constant_train_rate", "empirical_pitcher_prior", "hierarchical_prior_a50",
        "hierarchical_prior_a200", "hierarchical_prior_a1000", "official_random_forest",
        *[item["name"] for item in config["lightgbm_variants"]],
    ]
    primary = experiments[experiments["experiment_id"].isin(primary_ids)].drop_duplicates(
        "experiment_id", keep="last"
    )
    results = [
        {"model": row.experiment_id, "brier": float(row.raw_brier_score), "skill": float(row.local_brier_skill_score)}
        for row in primary.itertuples(index=False)
    ]
    calibration_ids = [
        "best_lgb_raw", "best_lgb_platt", "best_lgb_isotonic",
        "best_lgb_base_rate_shrink", "fixed_2023_oof_three_way_blend",
    ]
    cal_frame = experiments[experiments["experiment_id"].isin(calibration_ids)].drop_duplicates(
        "experiment_id", keep="last"
    )
    cal_rows = [
        {
            "model": row.experiment_id,
            "brier": float(row.calibrated_brier_score),
            "skill": float(row.local_brier_skill_score),
        }
        for row in cal_frame.itertuples(index=False)
    ]
    _render_findings(
        project_dir / "research" / "reports" / "initial_findings.md",
        results,
        cal_rows,
        best_name,
        int(params_logged["best_iteration"]),
        "best_lgb_raw",
        best_variant["feature_set"] == "trackman",
    )
    print(f"Final artifacts saved; peak RSS {monitor.peak_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("research/configs/default.json"))
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--num-rounds", type=int)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    if args.final_only:
        finalize_from_logged_results(project_dir, config_path, args.num_rounds)
    else:
        run(project_dir, config_path)


if __name__ == "__main__":
    main()
