from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.blend_experiment import _search_weights
from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, make_model, select_feature_columns


@dataclass(frozen=True)
class WeightSpec:
    name: str
    generation_fraction: float


def _weight_specs() -> list[WeightSpec]:
    return [
        WeightSpec("eligible_uniform", 0.0),
        WeightSpec("eligible_generation35", 0.35),
        WeightSpec("eligible_generation70", 0.70),
        WeightSpec("eligible_generation100", 1.0),
    ]


def _sample_weights(y: pd.Series, generation_fraction: float) -> np.ndarray:
    generation_weight = y.to_numpy(dtype=float) / float(y.mean())
    return (1.0 - generation_fraction) + generation_fraction * generation_weight


def _fit_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    spec: WeightSpec,
    seed: int,
) -> tuple[object, int]:
    model = make_model(seed)
    train_weight = _sample_weights(y_train, spec.generation_fraction)
    valid_weight = _sample_weights(y_valid, spec.generation_fraction)
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[valid_weight],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_)


def _fit_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    spec: WeightSpec,
    seed: int,
    n_estimators: int,
) -> object:
    model = make_model(seed, n_estimators=max(100, n_estimators))
    model.fit(
        X_train,
        y_train,
        sample_weight=_sample_weights(y_train, spec.generation_fraction),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_weighted_metric")
    parser.add_argument("--output", default="artifacts_weighted_metric/weighted_metric_member.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--n-iter", type=int, default=12000)
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
    if not submission.index.equals(X_test_all.index):
        raise ValueError("Test feature timestamps do not match sample submission.")

    specs = _weight_specs()
    valid_time = X_all.index >= pd.Timestamp(args.valid_start)
    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "weight_specs": [spec.__dict__ for spec in specs],
        "targets": {},
    }
    cache: dict[str, np.ndarray] = {
        "candidate_names": np.asarray([spec.name for spec in specs]),
        "test_index_ns": X_test_all.index.astype("int64").to_numpy(),
    }
    valid_predictions: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        columns = select_feature_columns(X_all, target, "base")
        X = X_all[columns]
        X_test = X_test_all.reindex(columns=columns)
        y = labels[target]
        valid = valid_time & y.notna()
        valid_eligible = valid & (y >= 0.10 * capacity)
        train_eligible = (~valid_time) & y.notna() & (y >= 0.10 * capacity)
        full_eligible = y.notna() & (y >= 0.10 * capacity)

        target_valid_preds = []
        target_test_preds = []
        target_report: dict[str, object] = {"candidate_metrics": {}}
        for spec_i, spec in enumerate(specs, start=1):
            seed = 24_000 + 100 * target_i + spec_i
            model, best_iteration = _fit_candidate(
                X.loc[train_eligible],
                y.loc[train_eligible],
                X.loc[valid_eligible],
                y.loc[valid_eligible],
                spec,
                seed,
            )
            raw_valid = np.clip(model.predict(X.loc[valid]), 0, capacity)
            scale, offset, calibrated_metric = calibrate(
                y.loc[valid].to_numpy(), raw_valid, capacity
            )
            pred_valid = np.clip(raw_valid * scale + offset, 0, capacity)

            final_model = _fit_final(
                X.loc[full_eligible],
                y.loc[full_eligible],
                spec,
                seed,
                best_iteration,
            )
            raw_test = np.clip(final_model.predict(X_test), 0, capacity)
            pred_test = np.clip(raw_test * scale + offset, 0, capacity)

            target_valid_preds.append(pred_valid)
            target_test_preds.append(pred_test)
            target_report["candidate_metrics"][spec.name] = {
                "best_iteration": int(best_iteration),
                "generation_fraction": spec.generation_fraction,
                "scale": float(scale),
                "offset": float(offset),
                "metric": calibrated_metric,
            }
            print(target, spec.name, calibrated_metric, flush=True)

        valid_matrix = np.column_stack(target_valid_preds)
        test_matrix = np.column_stack(target_test_preds)
        weights, blend_metric = _search_weights(
            valid_matrix,
            y.loc[valid].to_numpy(),
            capacity,
            seed=25_000 + target_i,
            n_iter=args.n_iter,
        )
        valid_blend = np.clip(valid_matrix @ weights, 0, capacity)
        test_blend = np.clip(test_matrix @ weights, 0, capacity)
        submission[target] = test_blend
        valid_predictions[target] = valid_blend
        valid_truth[target] = y.loc[valid].to_numpy()
        target_report["blend_metric"] = blend_metric
        target_report["selected_weights"] = {
            spec.name: float(weight)
            for spec, weight in zip(specs, weights)
            if weight > 1e-6
        }
        report["targets"][target] = target_report

        cache[f"{target}__valid_index_ns"] = X_all.index[valid].astype("int64").to_numpy()
        cache[f"{target}__valid_truth"] = y.loc[valid].to_numpy(dtype="float32")
        cache[f"{target}__valid_matrix"] = valid_matrix.astype("float32")
        cache[f"{target}__test_matrix"] = test_matrix.astype("float32")
        cache[f"{target}__selected_weights"] = weights.astype("float32")
        print(target, "BLEND", blend_metric, target_report["selected_weights"], flush=True)

    report["competition_metric"] = evaluate_competition(valid_truth, valid_predictions)
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "weighted_metric_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(artifact_dir / "prediction_cache.npz", **cache)

    print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
