from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import select_feature_columns


def _analog_columns(columns: list[str]) -> list[str]:
    keep = []
    for col in columns:
        if col in {
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "lead_hour",
        }:
            keep.append(col)
        elif col.startswith(("ldaps__ws", "gfs__ws")):
            keep.append(col)
        elif "hub_ws117" in col:
            keep.append(col)
        elif col.endswith(("surface_0_gust__mean", "surface_0_gust__max")):
            keep.append(col)
        elif col.endswith(("surface_0_sp__mean", "meanSea_0_prmsl__mean")):
            keep.append(col)
        elif col.endswith(("heightAboveGround_2_t__mean", "heightAboveGround_2_2t__mean")):
            keep.append(col)
    return keep


def _weighted_knn_predict(
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    X_query: pd.DataFrame,
    n_neighbors: int,
    power: float,
) -> np.ndarray:
    valid_ref = y_ref.notna()
    Xr = X_ref.loc[valid_ref]
    yr = y_ref.loc[valid_ref].to_numpy(dtype=float)
    medians = Xr.median()
    Xr = Xr.fillna(medians).fillna(0.0)
    X_query = X_query.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    Xr_scaled = scaler.fit_transform(Xr)
    Xq_scaled = scaler.transform(X_query)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(Xr)), metric="euclidean", algorithm="auto", n_jobs=-1)
    nn.fit(Xr_scaled)
    dist, idx = nn.kneighbors(Xq_scaled, return_distance=True)
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    return (yr[idx] * weights).sum(axis=1)


def _clip_predictions(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        target: np.clip(values, 0, CAPACITY_KWH[target])
        for target, values in preds.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_analog")
    parser.add_argument("--output", default="submissions/analog_v1.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--n-neighbors", type=int, default=80)
    parser.add_argument("--power", type=float, default=1.4)
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
    submission = sample.set_index(TIME_COL)

    base_cols = select_feature_columns(X_all, "kpx_group_1", "full")
    columns = _analog_columns(base_cols)
    X = X_all[columns]
    X_test = X_test_all.reindex(columns=columns)
    valid_time = X.index >= pd.Timestamp(args.valid_start)

    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "n_features": len(columns),
        "n_neighbors": args.n_neighbors,
        "power": args.power,
        "targets": {},
    }
    valid_pred: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for target, capacity in CAPACITY_KWH.items():
        y = labels[target]
        train_ref = (~valid_time) & y.notna()
        valid = valid_time & y.notna()
        full_ref = y.notna()
        pred_valid = np.clip(
            _weighted_knn_predict(X.loc[train_ref], y.loc[train_ref], X.loc[valid], args.n_neighbors, args.power),
            0,
            capacity,
        )
        metric = evaluate_group(y.loc[valid].to_numpy(), pred_valid, capacity)
        pred_test = np.clip(
            _weighted_knn_predict(X.loc[full_ref], y.loc[full_ref], X_test, args.n_neighbors, args.power),
            0,
            capacity,
        )
        submission[target] = pred_test
        valid_pred[target] = pred_valid
        valid_truth[target] = y.loc[valid].to_numpy()
        report["targets"][target] = {
            "train_ref_rows": int(train_ref.sum()),
            "full_ref_rows": int(full_ref.sum()),
            "valid_rows": int(valid.sum()),
            "metric": metric.to_dict(),
        }
        print(target, metric.to_dict(), flush=True)

    competition_metric = evaluate_competition(valid_truth, valid_pred)
    report["competition_metric"] = competition_metric
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "analog_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(competition_metric, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
