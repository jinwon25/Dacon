from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.scada_proxy_direct import _fit_predict_proxy
from experiments.scada_proxy_experiment import _hourly_scada
from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, select_feature_columns


def _target_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.045,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=35,
        l2_regularization=0.08,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )


def _fit_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_query: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    model = _target_model(seed)
    model.fit(X_train, y_train)
    return model.predict(X_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_scada_stack_hist")
    parser.add_argument("--output", default="submissions/scada_proxy_stack_hist.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
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
    valid_time = X.index >= pd.Timestamp(args.valid_start)

    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    scada = _hourly_scada(data_dir).reindex(X.index)

    proxy_train = pd.DataFrame(index=X.index)
    proxy_test = pd.DataFrame(index=X_test.index)
    proxy_report = {}
    for i, col in enumerate(scada.columns, start=1):
        y_scada = scada[col]
        pre_valid = (~valid_time) & y_scada.notna()
        full = y_scada.notna()
        pred_train = pd.Series(
            _fit_predict_proxy(X.loc[pre_valid], y_scada.loc[pre_valid], X, 17_000 + i, "hist"),
            index=X.index,
        )
        pred_test = _fit_predict_proxy(X.loc[full], y_scada.loc[full], X_test, 18_000 + i, "hist")
        name = f"proxy__{col}"
        proxy_train[name] = pred_train.astype("float32")
        proxy_test[name] = pred_test.astype("float32")
        valid_scada = valid_time & y_scada.notna()
        proxy_report[name] = {
            "pre_valid_rows": int(pre_valid.sum()),
            "full_rows": int(full.sum()),
            "valid_corr": float(np.corrcoef(pred_train.loc[valid_scada], y_scada.loc[valid_scada])[0, 1])
            if valid_scada.sum()
            else None,
        }
        print(name, proxy_report[name], flush=True)

    X_aug = X.join(proxy_train)
    X_test_aug = X_test.join(proxy_test)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)

    report: dict[str, object] = {"valid_start": args.valid_start, "proxy_report": proxy_report, "targets": {}}
    valid_pred: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = valid_time & y.notna()
        variants = {
            "all": (~valid_time) & y.notna(),
            "eligible_only": (~valid_time) & y.notna() & (y >= 0.10 * capacity),
        }
        best = {"score": -np.inf}
        for variant, train_mask in variants.items():
            raw_valid = np.clip(
                _fit_target(X_aug.loc[train_mask], y.loc[train_mask], X_aug.loc[valid], 19_000 + i),
                0,
                capacity,
            )
            scale, offset, metric = calibrate(y.loc[valid].to_numpy(), raw_valid, capacity)
            print(target, variant, metric, flush=True)
            if metric["score"] > best["score"]:
                best = {
                    "score": metric["score"],
                    "variant": variant,
                    "scale": scale,
                    "offset": offset,
                    "metric": metric,
                }

        final_mask = y.notna()
        if best["variant"] == "eligible_only":
            final_mask &= y >= 0.10 * capacity
        raw_test = np.clip(
            _fit_target(X_aug.loc[final_mask], y.loc[final_mask], X_test_aug, 20_000 + i),
            0,
            capacity,
        )
        submission[target] = np.clip(raw_test * best["scale"] + best["offset"], 0, capacity)
        valid_pred[target] = np.full(valid.sum(), np.nan)
        valid_truth[target] = y.loc[valid].to_numpy()
        report["targets"][target] = best | {"final_train_rows": int(final_mask.sum())}

    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "stack_hist_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["targets"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
