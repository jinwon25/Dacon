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
from src.metrics import CAPACITY_KWH, evaluate_competition
from train import calibrate, make_model, select_feature_columns


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    generation_fraction: float


SPECS = (
    CandidateSpec("global_uniform", 0.0),
    CandidateSpec("global_generation35", 0.35),
)

GROUP_META = {
    "kpx_group_1": {"group_id": 1.0, "manufacturer": 0.0, "turbine_count": 6.0, "rated_kw": 3600.0},
    "kpx_group_2": {"group_id": 2.0, "manufacturer": 0.0, "turbine_count": 6.0, "rated_kw": 3600.0},
    "kpx_group_3": {"group_id": 3.0, "manufacturer": 1.0, "turbine_count": 5.0, "rated_kw": 4200.0},
}


def _generic_own_columns(X: pd.DataFrame, target: str) -> list[str]:
    return [
        col
        for col in X.columns
        if f"__{target}__" in col and "hub_" not in col
    ]


def make_group_frame(X: pd.DataFrame, target: str) -> pd.DataFrame:
    base_cols = select_feature_columns(X, target, "base")
    out = X[base_cols].copy()
    own_cols = _generic_own_columns(X, target)
    own = X[own_cols].copy()
    own.columns = [col.replace(f"__{target}__", "__own_group__") for col in own.columns]
    out = out.join(own)
    for name, value in GROUP_META[target].items():
        out[name] = np.float32(value)
    return out.astype("float32")


def _stack_training_frames(
    frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    time_mask: np.ndarray,
    eligible_only: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X_parts = []
    y_parts = []
    group_parts = []
    for target, capacity in CAPACITY_KWH.items():
        y = labels[target]
        mask = time_mask & y.notna().to_numpy()
        if eligible_only:
            mask &= y.to_numpy() >= 0.10 * capacity
        X_parts.append(frames[target].loc[mask])
        y_parts.append(pd.Series(y.to_numpy()[mask] / capacity, index=frames[target].index[mask]))
        group_parts.append(pd.Series(target, index=frames[target].index[mask]))
    return (
        pd.concat(X_parts, axis=0, ignore_index=True),
        pd.concat(y_parts, axis=0, ignore_index=True),
        pd.concat(group_parts, axis=0, ignore_index=True),
    )


def _sample_weight(y_fraction: pd.Series, generation_fraction: float) -> np.ndarray | None:
    if generation_fraction <= 0:
        return None
    relative = y_fraction.to_numpy(dtype=float) / float(y_fraction.mean())
    return (1.0 - generation_fraction) + generation_fraction * relative


def _fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    spec: CandidateSpec,
    seed: int,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    model = make_model(seed, n_estimators=n_estimators or 1600)
    fit_args: dict[str, object] = {
        "sample_weight": _sample_weight(y_train, spec.generation_fraction),
        "callbacks": [lgb.log_evaluation(0)],
    }
    if X_valid is not None and y_valid is not None:
        fit_args.update(
            {
                "eval_set": [(X_valid, y_valid)],
                "eval_sample_weight": [
                    _sample_weight(y_valid, spec.generation_fraction)
                ] if spec.generation_fraction > 0 else None,
                "eval_metric": "l1",
                "callbacks": [lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
            }
        )
    model.fit(X_train, y_train, **fit_args)
    return model, int(model.best_iteration_ or model.n_estimators)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_global")
    parser.add_argument("--output", default="artifacts_global/global_member.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--n-iter", type=int, default=10_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("Building group-normalized feature frames...", flush=True)
    X_raw = build_features(data_dir, "train")
    X_test_raw = build_features(data_dir, "test")
    frames = {target: make_group_frame(X_raw, target) for target in CAPACITY_KWH}
    test_frames = {target: make_group_frame(X_test_raw, target) for target in CAPACITY_KWH}
    columns = list(frames["kpx_group_1"].columns)
    if any(list(frame.columns) != columns for frame in [*frames.values(), *test_frames.values()]):
        raise ValueError("Global group feature columns do not match.")

    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_raw.index)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)

    valid_time = X_raw.index >= pd.Timestamp(args.valid_start)
    X_train, y_train, _ = _stack_training_frames(
        frames, labels, np.asarray(~valid_time), eligible_only=True
    )
    X_valid_all, y_valid_all, _ = _stack_training_frames(
        frames, labels, np.asarray(valid_time), eligible_only=True
    )
    X_full, y_full, _ = _stack_training_frames(
        frames, labels, np.ones(len(X_raw), dtype=bool), eligible_only=True
    )

    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "n_features": len(columns),
        "train_rows": len(X_train),
        "valid_rows": len(X_valid_all),
        "full_rows": len(X_full),
        "specs": [spec.__dict__ for spec in SPECS],
        "targets": {target: {"candidates": {}} for target in CAPACITY_KWH},
    }
    cache: dict[str, np.ndarray] = {
        "candidate_names": np.asarray([spec.name for spec in SPECS]),
        "test_index_ns": X_test_raw.index.astype("int64").to_numpy(),
    }
    valid_members: dict[str, list[np.ndarray]] = {target: [] for target in CAPACITY_KWH}
    test_members: dict[str, list[np.ndarray]] = {target: [] for target in CAPACITY_KWH}

    for spec_i, spec in enumerate(SPECS, start=1):
        seed = 34_000 + spec_i
        model, best_iteration = _fit(
            X_train, y_train, X_valid_all, y_valid_all, spec, seed
        )
        final_model, _ = _fit(
            X_full, y_full, None, None, spec, seed, n_estimators=max(100, best_iteration)
        )
        for target, capacity in CAPACITY_KWH.items():
            y = labels[target]
            valid = valid_time & y.notna()
            raw_valid = np.clip(model.predict(frames[target].loc[valid]) * capacity, 0, capacity)
            scale, offset, metric = calibrate(y.loc[valid].to_numpy(), raw_valid, capacity)
            pred_valid = np.clip(raw_valid * scale + offset, 0, capacity)
            pred_test = np.clip(
                final_model.predict(test_frames[target]) * capacity * scale + offset,
                0,
                capacity,
            )
            valid_members[target].append(pred_valid)
            test_members[target].append(pred_test)
            report["targets"][target]["candidates"][spec.name] = {
                "best_iteration": best_iteration,
                "scale": scale,
                "offset": offset,
                "metric": metric,
            }
            print(target, spec.name, metric, flush=True)

    valid_truth: dict[str, np.ndarray] = {}
    valid_predictions: dict[str, np.ndarray] = {}
    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = valid_time & y.notna()
        truth = y.loc[valid].to_numpy()
        valid_matrix = np.column_stack(valid_members[target])
        test_matrix = np.column_stack(test_members[target])
        weights, metric = _search_weights(
            valid_matrix, truth, capacity, seed=35_000 + target_i, n_iter=args.n_iter
        )
        pred_valid = np.clip(valid_matrix @ weights, 0, capacity)
        pred_test = np.clip(test_matrix @ weights, 0, capacity)
        valid_truth[target] = truth
        valid_predictions[target] = pred_valid
        submission[target] = pred_test
        report["targets"][target]["blend_metric"] = metric
        report["targets"][target]["selected_weights"] = {
            spec.name: float(weight)
            for spec, weight in zip(SPECS, weights)
            if weight > 1e-8
        }
        cache[f"{target}__valid_index_ns"] = X_raw.index[valid].astype("int64").to_numpy()
        cache[f"{target}__valid_truth"] = truth.astype("float32")
        cache[f"{target}__valid_matrix"] = valid_matrix.astype("float32")
        cache[f"{target}__test_matrix"] = test_matrix.astype("float32")
        cache[f"{target}__selected_weights"] = weights.astype("float32")
        print(target, "BLEND", metric, report["targets"][target]["selected_weights"], flush=True)

    report["competition_metric"] = evaluate_competition(valid_truth, valid_predictions)
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "global_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(artifact_dir / "prediction_cache.npz", **cache)
    print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved member to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
