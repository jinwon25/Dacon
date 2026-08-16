"""Additional base learners for follow-up walk-forward experiments."""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data import TARGET_COL
from src.features import FeatureBuilder
from src.metrics import brier_score, brier_skill_score
from src.train import MemoryMonitor


RF_STRING_COLUMNS = ["top_bottom", "game_type", "base_state"]


def train_rf_variant_holdout(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    drop_columns = {"row_id", TARGET_COL, *config.get("drop_columns", ["season"])}
    features = [column for column in train.columns if column not in drop_columns]
    numeric = [column for column in features if column not in RF_STRING_COLUMNS]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                RF_STRING_COLUMNS,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(config.get("n_estimators", 300)),
                    max_depth=int(config.get("max_depth", 12)),
                    min_samples_leaf=int(config.get("min_samples_leaf", 200)),
                    max_features=config.get("max_features", "sqrt"),
                    n_jobs=int(config.get("thread_count", 6)),
                    random_state=int(config.get("seed", 42)),
                ),
            ),
        ]
    )
    target_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    target_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    with MemoryMonitor() as monitor:
        fit_started = time.perf_counter()
        model.fit(train.iloc[train_idx][features], target_train)
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        prediction = model.predict_proba(train.iloc[valid_idx][features])[:, 1]
        inference_seconds = time.perf_counter() - inference_started
        import pickle

        model_size_mb = len(
            pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        ) / (1024.0**2)
    return {
        "model": model,
        "prediction": np.asarray(prediction, dtype=np.float64),
        "target": target_valid,
        "brier": brier_score(target_valid, prediction),
        "skill": brier_skill_score(target_valid, prediction),
        "best_iteration": int(config.get("n_estimators", 300)),
        "build_seconds": 0.0,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": monitor.peak_mb,
        "feature_names": features,
    }


def _catboost_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical:
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    for column in out.columns:
        if column not in categorical:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    return out


def train_catboost_holdout(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    builder = FeatureBuilder(
        feature_set=config.get("feature_set", "hierarchical_v2"),
        drop_columns=list(config.get("drop_columns", ["season"])),
    )
    y_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder.fit(train.iloc[train_idx], y_train)
    with MemoryMonitor() as monitor:
        build_started = time.perf_counter()
        x_train = builder.raw_transform(train.iloc[train_idx])
        x_valid = builder.raw_transform(train.iloc[valid_idx])
        categorical = [
            column for column in builder.categorical_columns if column in x_train.columns
        ]
        x_train = _catboost_frame(x_train, categorical)
        x_valid = _catboost_frame(x_valid, categorical)
        train_pool = Pool(x_train, label=y_train, cat_features=categorical)
        valid_pool = Pool(x_valid, label=y_valid, cat_features=categorical)
        build_seconds = time.perf_counter() - build_started
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="BrierScore",
            iterations=int(config.get("iterations", 700)),
            depth=int(config.get("depth", 7)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            l2_leaf_reg=float(config.get("l2_leaf_reg", 5.0)),
            random_strength=float(config.get("random_strength", 0.5)),
            bootstrap_type=config.get("bootstrap_type", "Bayesian"),
            bagging_temperature=float(config.get("bagging_temperature", 1.0)),
            random_seed=int(config.get("seed", 42)),
            thread_count=int(config.get("thread_count", 6)),
            allow_writing_files=False,
            task_type="CPU",
            verbose=100,
            od_type="Iter",
            early_stopping_rounds=int(config.get("early_stopping_rounds", 80)),
        )
        fit_started = time.perf_counter()
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        prediction = model.predict_proba(valid_pool)[:, 1]
        inference_seconds = time.perf_counter() - inference_started
        model_size_mb = len(model._serialize_model()) / (1024.0**2)
        del x_train, x_valid, train_pool, valid_pool
        gc.collect()
    return {
        "builder": builder,
        "model": model,
        "prediction": np.asarray(prediction, dtype=np.float64),
        "target": y_valid,
        "brier": brier_score(y_valid, prediction),
        "skill": brier_skill_score(y_valid, prediction),
        "best_iteration": int(model.get_best_iteration()) + 1,
        "build_seconds": build_seconds,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": monitor.peak_mb,
        "feature_names": model.feature_names_,
        "categorical_columns": categorical,
    }


def train_xgboost_holdout(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    import xgboost as xgb

    builder = FeatureBuilder(
        feature_set=config.get("feature_set", "hierarchical_v2"),
        drop_columns=list(config.get("drop_columns", ["season"])),
    )
    y_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder.fit(train.iloc[train_idx], y_train)
    with MemoryMonitor() as monitor:
        build_started = time.perf_counter()
        x_train = builder.transform(train.iloc[train_idx])
        x_valid = builder.transform(train.iloc[valid_idx])
        dtrain = xgb.DMatrix(x_train, label=y_train, nthread=6)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, nthread=6)
        build_seconds = time.perf_counter() - build_started
        objective = config.get("objective", "binary:logistic")
        params = {
            "objective": objective,
            "eval_metric": "rmse",
            "tree_method": "hist",
            "max_depth": int(config.get("max_depth", 7)),
            "eta": float(config.get("learning_rate", 0.05)),
            "min_child_weight": float(config.get("min_child_weight", 100.0)),
            "lambda": float(config.get("lambda_l2", 5.0)),
            "subsample": float(config.get("subsample", 0.85)),
            "colsample_bytree": float(config.get("colsample_bytree", 0.9)),
            "seed": int(config.get("seed", 42)),
            "nthread": int(config.get("thread_count", 6)),
        }
        fit_started = time.perf_counter()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=int(config.get("iterations", 700)),
            evals=[(dvalid, "valid")],
            early_stopping_rounds=int(config.get("early_stopping_rounds", 80)),
            verbose_eval=100,
        )
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        prediction = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        inference_seconds = time.perf_counter() - inference_started
        model_size_mb = len(model.save_raw()) / (1024.0**2)
        del x_train, x_valid, dtrain, dvalid
        gc.collect()
    return {
        "builder": builder,
        "model": model,
        "prediction": np.asarray(prediction, dtype=np.float64),
        "target": y_valid,
        "brier": brier_score(y_valid, prediction),
        "skill": brier_skill_score(y_valid, prediction),
        "best_iteration": int(model.best_iteration) + 1,
        "build_seconds": build_seconds,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": monitor.peak_mb,
        "feature_names": builder.transform(train.iloc[valid_idx[:1]]).columns.tolist(),
    }
