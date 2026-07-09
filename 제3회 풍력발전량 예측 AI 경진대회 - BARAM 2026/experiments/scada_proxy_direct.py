from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.scada_proxy_experiment import _hourly_scada
from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, select_feature_columns


def _make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.055,
        max_iter=420,
        max_leaf_nodes=31,
        min_samples_leaf=35,
        l2_regularization=0.05,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
    )


def _make_lgbm_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=35,
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


def _fit_predict_proxy(
    X_train: pd.DataFrame,
    y_scada: pd.Series,
    X_query: pd.DataFrame,
    seed: int,
    model_type: str,
) -> np.ndarray:
    mask = y_scada.notna()
    if model_type == "hist":
        model = _make_model(seed)
        model.fit(X_train.loc[mask], y_scada.loc[mask])
    elif model_type == "lgbm":
        model = _make_lgbm_model(seed)
        model.fit(X_train.loc[mask], y_scada.loc[mask], callbacks=[lgb.log_evaluation(0)])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.predict(X_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_scada_direct")
    parser.add_argument("--output", default="submissions/scada_proxy_direct_v1.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--model-type", choices=["hist", "lgbm"], default="hist")
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
    scada = _hourly_scada(data_dir).reindex(X.index)

    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)

    valid_time = X.index >= pd.Timestamp(args.valid_start)
    report: dict[str, object] = {"valid_start": args.valid_start, "targets": {}}
    valid_pred: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        y_scada = scada[f"{target}__scada_power_sum"]
        train_mask = (~valid_time) & y_scada.notna()
        valid_mask = valid_time & y.notna()

        proxy_valid = _fit_predict_proxy(
            X.loc[train_mask],
            y_scada.loc[train_mask],
            X.loc[valid_mask],
            15_000 + i,
            args.model_type,
        )
        proxy_valid = np.clip(proxy_valid, 0, capacity)
        scale, offset, metric = calibrate(y.loc[valid_mask].to_numpy(), proxy_valid, capacity)
        pred_valid = np.clip(proxy_valid * scale + offset, 0, capacity)
        fair_metric = evaluate_group(y.loc[valid_mask].to_numpy(), pred_valid, capacity)

        full_proxy_mask = y_scada.notna()
        proxy_test = _fit_predict_proxy(
            X.loc[full_proxy_mask],
            y_scada.loc[full_proxy_mask],
            X_test,
            16_000 + i,
            args.model_type,
        )
        pred_test = np.clip(proxy_test * scale + offset, 0, capacity)

        submission[target] = pred_test
        valid_pred[target] = pred_valid
        valid_truth[target] = y.loc[valid_mask].to_numpy()
        report["targets"][target] = {
            "proxy_train_rows": int(train_mask.sum()),
            "proxy_full_rows": int(full_proxy_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "model_type": args.model_type,
            "scale": float(scale),
            "offset": float(offset),
            "metric": fair_metric.to_dict(),
        }
        print(target, report["targets"][target], flush=True)

    report["competition_metric"] = evaluate_competition(valid_truth, valid_pred)
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "scada_direct_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
