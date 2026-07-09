from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, make_catboost_model, make_model, select_feature_columns


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    feature_set: str
    train_variant: str


def _fit_candidate(
    spec: CandidateSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> tuple[object, int]:
    if spec.family == "lgbm":
        model = make_model(seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return model, int(model.best_iteration_)
    if spec.family == "catboost":
        model = make_catboost_model(seed)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        return model, int(model.get_best_iteration() + 1)
    if spec.family == "xgboost":
        model = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=1400,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=100,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return model, int(model.best_iteration + 1)
    raise ValueError(f"Unknown family: {spec.family}")


def _fit_final(spec: CandidateSpec, X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_estimators: int) -> object:
    if spec.family == "lgbm":
        model = make_model(seed, n_estimators=max(100, n_estimators))
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
        return model
    if spec.family == "catboost":
        model = make_catboost_model(seed, iterations=max(100, n_estimators))
        model.fit(X_train, y_train)
        return model
    if spec.family == "xgboost":
        model = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=max(100, n_estimators),
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train, verbose=False)
        return model
    raise ValueError(f"Unknown family: {spec.family}")


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("lgbm_base_all", "lgbm", "base", "all"),
        CandidateSpec("lgbm_base_eligible", "lgbm", "base", "eligible_only"),
        CandidateSpec("cat_base_all", "catboost", "base", "all"),
        CandidateSpec("cat_base_eligible", "catboost", "base", "eligible_only"),
        CandidateSpec("xgb_base_all", "xgboost", "base", "all"),
        CandidateSpec("xgb_base_eligible", "xgboost", "base", "eligible_only"),
        CandidateSpec("lgbm_own_idw_nohub_all", "lgbm", "own_idw_nohub", "all"),
        CandidateSpec("lgbm_own_idw_nohub_eligible", "lgbm", "own_idw_nohub", "eligible_only"),
    ]


def _search_weights(preds: np.ndarray, y_true: np.ndarray, capacity: float, seed: int, n_iter: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_models = preds.shape[1]
    candidates: list[np.ndarray] = []
    candidates.extend(np.eye(n_models))
    candidates.append(np.full(n_models, 1.0 / n_models))

    # Sparse blends usually generalize better than dense blends on one-year validation.
    for active in range(2, min(4, n_models) + 1):
        for _ in range(max(200, n_iter // 8)):
            idx = rng.choice(n_models, size=active, replace=False)
            w = np.zeros(n_models)
            w[idx] = rng.dirichlet(np.ones(active))
            candidates.append(w)
    for _ in range(n_iter):
        alpha = rng.uniform(0.25, 2.0, size=n_models)
        candidates.append(rng.dirichlet(alpha))

    best_score = -np.inf
    best_w = candidates[0]
    best_metric = None
    for w in candidates:
        blended = np.clip(preds @ w, 0, capacity)
        metric = evaluate_group(y_true, blended, capacity)
        if metric.score > best_score:
            best_score = metric.score
            best_w = w.copy()
            best_metric = metric
    return best_w, best_metric.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_blend")
    parser.add_argument("--output", default="submissions/blend_v1.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--n-iter", type=int, default=8000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building features...", flush=True)
    X_all = build_features(data_dir, "train")
    X_test_all = build_features(data_dir, "test")
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_all.index)

    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    indexed_submission = sample.set_index(TIME_COL)
    if not indexed_submission.index.equals(X_test_all.index):
        raise ValueError("Test feature timestamps do not match sample submission.")

    valid_time = X_all.index >= pd.Timestamp(args.valid_start)
    specs = _candidate_specs()
    report: dict[str, object] = {"valid_start": args.valid_start, "candidates": [s.__dict__ for s in specs], "targets": {}}
    valid_predictions: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid_mask = valid_time & y.notna()
        valid_truth[target] = y.loc[valid_mask].to_numpy()
        target_valid_preds = []
        target_test_preds = []
        target_report = {"candidate_metrics": {}, "selected_weights": {}}

        for spec_i, spec in enumerate(specs, start=1):
            cols = select_feature_columns(X_all, target, spec.feature_set)
            X = X_all[cols]
            X_test = X_test_all.reindex(columns=cols)
            train_mask = (~valid_time) & y.notna()
            if spec.train_variant == "eligible_only":
                train_mask &= y >= 0.10 * capacity

            seed = 7000 + 100 * target_i + spec_i
            model, best_iteration = _fit_candidate(
                spec,
                X.loc[train_mask],
                y.loc[train_mask],
                X.loc[valid_mask],
                y.loc[valid_mask],
                seed=seed,
            )
            raw_valid = np.clip(model.predict(X.loc[valid_mask]), 0, capacity)
            scale, offset, calibrated_metric = calibrate(y.loc[valid_mask].to_numpy(), raw_valid, capacity)
            calibrated_valid = np.clip(raw_valid * scale + offset, 0, capacity)

            final_train_mask = y.notna()
            if spec.train_variant == "eligible_only":
                final_train_mask &= y >= 0.10 * capacity
            final_model = _fit_final(spec, X.loc[final_train_mask], y.loc[final_train_mask], seed, best_iteration)
            raw_test = np.clip(final_model.predict(X_test), 0, capacity)
            calibrated_test = np.clip(raw_test * scale + offset, 0, capacity)

            target_valid_preds.append(calibrated_valid)
            target_test_preds.append(calibrated_test)
            target_report["candidate_metrics"][spec.name] = {
                "best_iteration": int(best_iteration),
                "n_features": len(cols),
                "train_rows": int(train_mask.sum()),
                "final_train_rows": int(final_train_mask.sum()),
                "scale": float(scale),
                "offset": float(offset),
                "metric": calibrated_metric,
            }
            print(target, spec.name, calibrated_metric, flush=True)

        pred_matrix = np.column_stack(target_valid_preds)
        weights, blend_metric = _search_weights(
            pred_matrix,
            valid_truth[target],
            capacity,
            seed=9000 + target_i,
            n_iter=args.n_iter,
        )
        test_matrix = np.column_stack(target_test_preds)
        indexed_submission[target] = np.clip(test_matrix @ weights, 0, capacity)
        valid_predictions[target] = np.clip(pred_matrix @ weights, 0, capacity)
        target_report["blend_metric"] = blend_metric
        target_report["selected_weights"] = {
            spec.name: float(weight)
            for spec, weight in zip(specs, weights)
            if weight > 1e-6
        }
        report["targets"][target] = target_report
        print(target, "BLEND", blend_metric, target_report["selected_weights"], flush=True)

    competition_metric = evaluate_competition(valid_truth, valid_predictions)
    report["competition_metric"] = competition_metric

    output = indexed_submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "blend_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(report, artifact_dir / "blend_report.joblib")

    print(json.dumps(competition_metric, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
