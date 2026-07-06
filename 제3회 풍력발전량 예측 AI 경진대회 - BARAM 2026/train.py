from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

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
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building train features...", flush=True)
    X = build_features(args.data_dir, "train")
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)

    valid_time = X.index >= pd.Timestamp(args.valid_start)
    report: dict[str, object] = {"valid_start": args.valid_start, "n_features": X.shape[1], "targets": {}}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        base_train = (~valid_time) & y.notna()
        valid = valid_time & y.notna()
        variants = {
            "all": base_train,
            "eligible_only": base_train & (y >= 0.10 * capacity),
        }
        variant_results = {}
        best_variant = None
        best_score = -np.inf
        best_iteration = 1200
        best_calibration = (1.0, 0.0)

        for variant, train_mask in variants.items():
            model = make_model(2026 + i)
            model.fit(
                X.loc[train_mask],
                y.loc[train_mask],
                eval_set=[(X.loc[valid], y.loc[valid])],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            raw_pred = np.clip(model.predict(X.loc[valid]), 0, capacity)
            raw_metric = evaluate_group(y.loc[valid].to_numpy(), raw_pred, capacity)
            scale, offset, calibrated_metric = calibrate(y.loc[valid].to_numpy(), raw_pred, capacity)
            variant_results[variant] = {
                "train_rows": int(train_mask.sum()),
                "best_iteration": int(model.best_iteration_),
                "raw": raw_metric.to_dict(),
                "calibration": {"scale": scale, "offset": offset, "metric": calibrated_metric},
            }
            print(target, variant, variant_results[variant], flush=True)
            if calibrated_metric["score"] > best_score:
                best_score = calibrated_metric["score"]
                best_variant = variant
                best_iteration = int(model.best_iteration_)
                best_calibration = (scale, offset)

        full_mask = y.notna()
        if best_variant == "eligible_only":
            full_mask &= y >= 0.10 * capacity
        final_model = make_model(2026 + i, n_estimators=max(100, best_iteration))
        final_model.fit(X.loc[full_mask], y.loc[full_mask], callbacks=[lgb.log_evaluation(0)])
        joblib.dump(final_model, artifact_dir / f"{target}.joblib")

        report["targets"][target] = {
            "selected_variant": best_variant,
            "full_train_rows": int(full_mask.sum()),
            "best_iteration": max(100, best_iteration),
            "scale": best_calibration[0],
            "offset": best_calibration[1],
            "validation": variant_results,
        }

    joblib.dump(list(X.columns), artifact_dir / "feature_columns.joblib")
    (artifact_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved models and report to {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
