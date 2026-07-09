from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import build_features
from src.metrics import CAPACITY_KWH, evaluate_group


def make_model(seed: int, n_estimators: int = 1200) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def make_catboost_model(seed: int, iterations: int = 2500) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=150,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def select_feature_columns(X: pd.DataFrame, target: str, feature_set: str) -> list[str]:
    base_cols = [c for c in X.columns if "hub_" not in c and "__kpx_group_" not in c]
    if feature_set == "base":
        return base_cols
    if feature_set == "full":
        return list(X.columns)
    if feature_set == "own_idw":
        return base_cols + [c for c in X.columns if f"__{target}__" in c]
    if feature_set == "own_idw_nohub":
        return base_cols + [c for c in X.columns if f"__{target}__" in c and "hub_" not in c]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def fit_candidate(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> tuple[object, int]:
    if family == "lgbm":
        model = make_model(seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return model, int(model.best_iteration_)
    if family == "catboost":
        model = make_catboost_model(seed)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        return model, int(model.get_best_iteration() + 1)
    raise ValueError(f"Unknown model family: {family}")


def fit_final(family: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_estimators: int) -> object:
    if family == "lgbm":
        model = make_model(seed, n_estimators=max(100, n_estimators))
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
        return model
    if family == "catboost":
        model = make_catboost_model(seed, iterations=max(100, n_estimators))
        model.fit(X_train, y_train)
        return model
    raise ValueError(f"Unknown model family: {family}")


def calibrate(y_true: np.ndarray, raw_pred: np.ndarray, capacity: float) -> tuple[float, float, dict]:
    best = (-np.inf, 1.0, 0.0, None)
    for scale in np.arange(0.96, 1.041, 0.01):
        for offset in np.arange(-600.0, 601.0, 100.0):
            pred = np.clip(raw_pred * scale + offset, 0, capacity)
            metric = evaluate_group(y_true, pred, capacity)
            if metric.score > best[0]:
                best = (metric.score, float(scale), float(offset), metric)
    return best[1], best[2], best[3].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument(
        "--feature-set",
        choices=["base", "full", "own_idw", "own_idw_nohub"],
        default="base",
        help="Feature subset to train on. 'base' reproduces the first public submission feature surface.",
    )
    parser.add_argument(
        "--catboost-targets",
        default="",
        help="Comma-separated targets to evaluate with CatBoost in addition to LightGBM, or '__all__'.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building train features...", flush=True)
    X = build_features(args.data_dir, "train")
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)

    valid_time = X.index >= pd.Timestamp(args.valid_start)
    catboost_targets = {x.strip() for x in args.catboost_targets.split(",") if x.strip()}
    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "feature_set": args.feature_set,
        "raw_n_features": X.shape[1],
        "targets": {},
    }
    feature_columns: dict[str, list[str]] = {}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        columns = select_feature_columns(X, target, args.feature_set)
        feature_columns[target] = columns
        Xt = X[columns]
        y = labels[target]
        base_train = (~valid_time) & y.notna()
        valid = valid_time & y.notna()
        variants = {
            "all": base_train,
            "eligible_only": base_train & (y >= 0.10 * capacity),
        }
        variant_results = {}
        candidate_families = ["lgbm"]
        if "__all__" in catboost_targets or target in catboost_targets:
            candidate_families.append("catboost")

        best_family = None
        best_variant = None
        best_score = -np.inf
        best_iteration = 1200
        best_calibration = (1.0, 0.0)

        for family in candidate_families:
            for variant, train_mask in variants.items():
                model, iteration = fit_candidate(
                    family,
                    Xt.loc[train_mask],
                    y.loc[train_mask],
                    Xt.loc[valid],
                    y.loc[valid],
                    seed=2026 + i + (1000 if family == "catboost" else 0),
                )
                raw_pred = np.clip(model.predict(Xt.loc[valid]), 0, capacity)
                raw_metric = evaluate_group(y.loc[valid].to_numpy(), raw_pred, capacity)
                scale, offset, calibrated_metric = calibrate(y.loc[valid].to_numpy(), raw_pred, capacity)
                key = f"{family}:{variant}"
                variant_results[key] = {
                    "family": family,
                    "variant": variant,
                    "train_rows": int(train_mask.sum()),
                    "best_iteration": int(iteration),
                    "raw": raw_metric.to_dict(),
                    "calibration": {"scale": scale, "offset": offset, "metric": calibrated_metric},
                }
                print(target, key, variant_results[key], flush=True)
                if calibrated_metric["score"] > best_score:
                    best_score = calibrated_metric["score"]
                    best_family = family
                    best_variant = variant
                    best_iteration = int(iteration)
                    best_calibration = (scale, offset)

        full_mask = y.notna()
        if best_variant == "eligible_only":
            full_mask &= y >= 0.10 * capacity
        final_model = fit_final(
            best_family,
            Xt.loc[full_mask],
            y.loc[full_mask],
            seed=2026 + i + (1000 if best_family == "catboost" else 0),
            n_estimators=best_iteration,
        )
        joblib.dump(final_model, artifact_dir / f"{target}.joblib")

        report["targets"][target] = {
            "selected_family": best_family,
            "selected_variant": best_variant,
            "full_train_rows": int(full_mask.sum()),
            "best_iteration": max(100, best_iteration),
            "n_features": len(columns),
            "scale": best_calibration[0],
            "offset": best_calibration[1],
            "validation": variant_results,
        }

    joblib.dump(feature_columns, artifact_dir / "feature_columns.joblib")
    (artifact_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved models and report to {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
