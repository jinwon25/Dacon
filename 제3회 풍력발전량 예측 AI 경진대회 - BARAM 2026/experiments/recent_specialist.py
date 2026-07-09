from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, make_catboost_model, make_model, select_feature_columns


def _fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> tuple[object, int]:
    model = make_model(seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_)


def _fit_cat(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> tuple[object, int]:
    model = make_catboost_model(seed)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    return model, int(model.get_best_iteration() + 1)


def _fit_final(family: str, X: pd.DataFrame, y: pd.Series, seed: int, n_estimators: int) -> object:
    if family == "lgbm":
        model = make_model(seed, n_estimators=max(100, n_estimators))
        model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        return model
    if family == "catboost":
        model = make_catboost_model(seed, iterations=max(100, n_estimators))
        model.fit(X, y)
        return model
    raise ValueError(family)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_recent")
    parser.add_argument("--output", default="submissions/recent_specialist_v1.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--valid-train-start", default="2023-01-01 00:00:00")
    parser.add_argument("--final-train-start", default="2024-01-01 00:00:00")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building features...", flush=True)
    X_all = build_features(data_dir, "train")
    X_test_all = build_features(data_dir, "test")
    cols = select_feature_columns(X_all, "kpx_group_1", "base")
    X = X_all[cols]
    X_test = X_test_all.reindex(columns=cols)

    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)

    valid_start = pd.Timestamp(args.valid_start)
    valid_train_start = pd.Timestamp(args.valid_train_start)
    final_train_start = pd.Timestamp(args.final_train_start)

    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "valid_train_start": args.valid_train_start,
        "final_train_start": args.final_train_start,
        "targets": {},
    }
    valid_pred: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = (X.index >= valid_start) & y.notna()
        base_train = (X.index >= valid_train_start) & (X.index < valid_start) & y.notna()
        variants = {
            "all": base_train,
            "eligible_only": base_train & (y >= 0.10 * capacity),
        }
        families = ["lgbm", "catboost"] if target == "kpx_group_3" else ["lgbm"]
        best = {"score": -np.inf}

        for family in families:
            for variant, train_mask in variants.items():
                if train_mask.sum() < 500:
                    continue
                if family == "lgbm":
                    model, iteration = _fit_lgbm(X.loc[train_mask], y.loc[train_mask], X.loc[valid], y.loc[valid], 12000 + i)
                else:
                    model, iteration = _fit_cat(X.loc[train_mask], y.loc[train_mask], X.loc[valid], y.loc[valid], 13000 + i)
                raw = np.clip(model.predict(X.loc[valid]), 0, capacity)
                scale, offset, metric = calibrate(y.loc[valid].to_numpy(), raw, capacity)
                print(target, family, variant, metric, flush=True)
                if metric["score"] > best["score"]:
                    best = {
                        "score": metric["score"],
                        "family": family,
                        "variant": variant,
                        "best_iteration": iteration,
                        "scale": scale,
                        "offset": offset,
                        "valid_pred": np.clip(raw * scale + offset, 0, capacity),
                        "metric": metric,
                    }

        final_mask = (X.index >= final_train_start) & y.notna()
        if best["variant"] == "eligible_only":
            final_mask &= y >= 0.10 * capacity
        final_model = _fit_final(
            best["family"],
            X.loc[final_mask],
            y.loc[final_mask],
            seed=14000 + i,
            n_estimators=int(best["best_iteration"]),
        )
        submission[target] = np.clip(final_model.predict(X_test) * best["scale"] + best["offset"], 0, capacity)
        valid_pred[target] = best["valid_pred"]
        valid_truth[target] = y.loc[valid].to_numpy()
        report["targets"][target] = {
            k: v for k, v in best.items() if k != "valid_pred"
        } | {"final_train_rows": int(final_mask.sum())}

    report["competition_metric"] = evaluate_competition(valid_truth, valid_pred)
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "recent_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
