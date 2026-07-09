from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group


def _target_feature_columns(X: pd.DataFrame, target: str) -> list[str]:
    tokens = (
        f"__{target}__hub_ws117__idw",
        f"__{target}__hub_u117__idw",
        f"__{target}__hub_v117__idw",
        f"__{target}__hub_dir_sin__idw",
        f"__{target}__hub_dir_cos__idw",
        f"__{target}__ws",
        f"__{target}__surface_0_gust__idw",
        f"__{target}__heightAboveGround_2",
        f"__{target}__surface_0_sp__idw",
    )
    own_cols = [c for c in X.columns if any(token in c for token in tokens)]
    calendar_cols = ["hour", "month", "dayofweek", "lead_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    summary_cols = [
        c
        for c in X.columns
        if (
            ("hub_ws117" in c or "surface_0_gust" in c or c.endswith("__ws10__mean"))
            and c.endswith(("__mean", "__std", "__min", "__max"))
        )
    ]
    cols = [c for c in [*own_cols, *summary_cols, *calendar_cols] if c in X.columns]
    return list(dict.fromkeys(cols))


def _wind_index(X: pd.DataFrame, target: str) -> np.ndarray:
    preferred = [c for c in X.columns if c.endswith(f"__{target}__hub_ws117__idw")]
    fallback = [
        c
        for c in X.columns
        if f"__{target}__" in c and c.endswith("__idw") and ("ws" in c or "gust" in c)
    ]
    cols = preferred or fallback
    if not cols:
        raise ValueError(f"No target wind-speed feature found for {target}.")
    return X[cols].mean(axis=1).to_numpy(dtype=float)


def _make_residual_model(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=550,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.03,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
        ),
    )


def _fit_predict_one(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    target: str,
    capacity: float,
    valid_time: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cols = _target_feature_columns(X, target)
    wind = _wind_index(X, target)
    wind_test = _wind_index(X_test, target)

    train_mask = (~valid_time) & y.notna()
    valid_mask = valid_time & y.notna()

    y_norm = (y / capacity).clip(0.0, 1.0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(wind[train_mask], y_norm.loc[train_mask].to_numpy())
    iso_valid = iso.predict(wind[valid_mask])
    residual = y_norm.loc[train_mask].to_numpy() - iso.predict(wind[train_mask])

    residual_model = _make_residual_model(seed)
    residual_model.fit(X.loc[train_mask, cols], residual)
    pred_valid_norm = np.clip(iso_valid + residual_model.predict(X.loc[valid_mask, cols]), 0.0, 1.0)
    pred_valid = pred_valid_norm * capacity
    metric = evaluate_group(y.loc[valid_mask].to_numpy(), pred_valid, capacity)

    final_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    full_mask = y.notna()
    final_iso.fit(wind[full_mask], y_norm.loc[full_mask].to_numpy())
    final_residual = y_norm.loc[full_mask].to_numpy() - final_iso.predict(wind[full_mask])
    final_model = _make_residual_model(seed + 1000)
    final_model.fit(X.loc[full_mask, cols], final_residual)
    pred_test_norm = np.clip(final_iso.predict(wind_test) + final_model.predict(X_test[cols]), 0.0, 1.0)
    pred_test = pred_test_norm * capacity

    report = {
        "n_features": len(cols),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "full_rows": int(full_mask.sum()),
        "metric": metric.to_dict(),
        "feature_sample": cols[:25],
    }
    return pred_valid, pred_test, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_power_curve")
    parser.add_argument("--output", default="submissions/power_curve_residual.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building features...", flush=True)
    X = build_features(data_dir, "train")
    X_test = build_features(data_dir, "test")
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)

    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    output = sample.set_index(TIME_COL)
    if not output.index.equals(X_test.index):
        raise ValueError("Test feature timestamps do not match sample submission.")

    valid_time = np.asarray(X.index >= pd.Timestamp(args.valid_start))
    valid_predictions: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}
    report: dict[str, object] = {"valid_start": args.valid_start, "targets": {}}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        pred_valid, pred_test, target_report = _fit_predict_one(
            X,
            X_test,
            labels[target],
            target,
            capacity,
            valid_time,
            seed=202607 + i,
        )
        valid_mask = valid_time & labels[target].notna().to_numpy()
        valid_predictions[target] = pred_valid
        valid_truth[target] = labels[target].loc[valid_mask].to_numpy()
        output[target] = np.clip(pred_test, 0, capacity)
        report["targets"][target] = target_report
        print(target, json.dumps(target_report["metric"], ensure_ascii=False), flush=True)

    competition_metric = evaluate_competition(valid_truth, valid_predictions)
    report["competition_metric"] = competition_metric

    result = output.reset_index()[sample.columns]
    result[TIME_COL] = result[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "power_curve_residual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(competition_metric, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
