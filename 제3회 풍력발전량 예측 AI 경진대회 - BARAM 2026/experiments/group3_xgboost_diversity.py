from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from experiments.group3_physical_catboost import (
    CAPACITY,
    H2_START,
    Q2_START,
    TARGET,
    _bootstrap_days,
    _compare,
)
from src.metrics import evaluate_group
from train import calibrate, select_feature_columns


def _model(seed: int, n_estimators: int = 1_600) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=n_estimators,
        learning_rate=0.025,
        max_depth=7,
        min_child_weight=12.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=5.0,
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=120 if n_estimators >= 1_600 else None,
    )


def _fit_select(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    iterations: list[int] = []
    train_rows = np.flatnonzero(train_mask)
    valid_rows = np.flatnonzero(valid_mask)
    for seed in seeds:
        model = _model(seed)
        model.fit(
            X.iloc[train_rows],
            y.iloc[train_rows],
            eval_set=[(X.iloc[valid_rows], y.iloc[valid_rows])],
            verbose=False,
        )
        predictions.append(
            np.clip(model.predict(X.iloc[valid_rows]), 0.0, CAPACITY)
        )
        try:
            best_iteration = int(model.best_iteration + 1)
        except AttributeError:
            best_iteration = int(model.n_estimators)
        iterations.append(max(100, best_iteration))
    return np.mean(predictions, axis=0), iterations, predictions


def _fit_locked(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
    iterations: list[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    train_rows = np.flatnonzero(train_mask)
    valid_rows = np.flatnonzero(valid_mask)
    for seed, n_estimators in zip(seeds, iterations, strict=True):
        model = _model(seed, n_estimators)
        model.fit(X.iloc[train_rows], y.iloc[train_rows], verbose=False)
        predictions.append(
            np.clip(model.predict(X.iloc[valid_rows]), 0.0, CAPACITY)
        )
    return np.mean(predictions, axis=0), predictions


def _exact_current_and_truth(
    driver: np.lib.npyio.NpzFile,
    meta: np.lib.npyio.NpzFile,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    meta_index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    truth = (
        pd.Series(driver[f"{TARGET}__valid_truth"].astype(float), index=driver_index)
        .reindex(index)
        .to_numpy(dtype=float)
    )
    current = (
        pd.Series(meta["valid_candidate"].astype(float), index=meta_index)
        .reindex(index)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(truth).all() or not np.isfinite(current).all():
        raise ValueError("Exact lineage does not cover requested index")
    return truth, current


def run_experiment(
    feature_cache: Path,
    labels_path: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
    parent_cache_path: Path,
    artifact_dir: Path,
    evaluation_output: Path,
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    features = pd.read_pickle(feature_cache)
    columns = select_feature_columns(features, TARGET, "base")
    X = features[columns]
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(features.index)
    y = labels[TARGET]
    eligible = y.notna() & (y >= 0.10 * CAPACITY)
    q2_train = np.asarray(
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < Q2_START)
        & eligible
    )
    q2_valid = np.asarray(
        (features.index >= Q2_START) & (features.index < H2_START) & y.notna()
    )
    h2_train = np.asarray(
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < H2_START)
        & eligible
    )
    h2_valid = np.asarray((features.index >= H2_START) & y.notna())

    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    q2_index = features.index[q2_valid]
    h2_index = features.index[h2_valid]
    q2_truth, q2_current = _exact_current_and_truth(driver, meta, q2_index)
    h2_truth, h2_current = _exact_current_and_truth(driver, meta, h2_index)

    q2_mean, iterations, q2_seed_raw = _fit_select(
        X, y, q2_train, q2_valid, seeds
    )
    scale, offset, _ = calibrate(q2_truth, q2_mean, CAPACITY)
    q2_seed = [
        np.clip(prediction * scale + offset, 0.0, CAPACITY)
        for prediction in q2_seed_raw
    ]
    q2_prediction = np.clip(q2_mean * scale + offset, 0.0, CAPACITY)

    h2_mean, h2_seed_raw = _fit_locked(
        X, y, h2_train, h2_valid, seeds, iterations
    )
    h2_seed = [
        np.clip(prediction * scale + offset, 0.0, CAPACITY)
        for prediction in h2_seed_raw
    ]
    h2_prediction = np.clip(h2_mean * scale + offset, 0.0, CAPACITY)
    q2_comparison = _compare(q2_truth, q2_current, q2_prediction)
    h2_comparison = _compare(h2_truth, h2_current, h2_prediction)

    parent_comparison: dict[str, object] | None = None
    parent_correlation: float | None = None
    if parent_cache_path.exists():
        parent = np.load(parent_cache_path, allow_pickle=True)
        parent_index = pd.DatetimeIndex(pd.to_datetime(parent["h2_index_ns"]))
        parent_prediction = (
            pd.Series(parent["h2_base_control"].astype(float), index=parent_index)
            .reindex(h2_index)
            .to_numpy(dtype=float)
        )
        if np.isfinite(parent_prediction).all():
            parent_comparison = _compare(
                h2_truth, parent_prediction, h2_prediction
            )
            parent_correlation = float(
                np.corrcoef(parent_prediction, h2_prediction)[0, 1]
            )

    seed_deltas = [
        {"seed": seed, "delta": _compare(h2_truth, h2_current, prediction)["delta"]}
        for seed, prediction in zip(seeds, h2_seed, strict=True)
    ]
    monthly = {
        str(month): _compare(
            h2_truth[h2_index.month == month],
            h2_current[h2_index.month == month],
            h2_prediction[h2_index.month == month],
        )["delta"]
        for month in range(7, 13)
    }
    bootstrap = _bootstrap_days(
        h2_truth, h2_current, h2_prediction, h2_index, n_bootstrap
    )
    movement = h2_prediction - h2_current
    duration = time.perf_counter() - started

    report: dict[str, object] = {
        "method": "eligible-only XGBoost absolute-error diversity member",
        "hypothesis": (
            "XGBoost's regularized depth-wise histogram trees add useful error diversity "
            "to the LightGBM/CatBoost lineage without changing the input surface."
        ),
        "n_features": len(columns),
        "seeds": list(seeds),
        "train_contract": {
            "selection": "2023-01-01 through 2024-Q1 -> 2024-Q2",
            "locked": "2023-01-01 through 2024-H1 -> 2024-H2",
            "eligible_training_only": True,
            "objective": "reg:absoluteerror",
        },
        "selection_q2": {
            "best_iterations": iterations,
            "calibration": {"scale": scale, "offset": offset},
            "vs_current": q2_comparison,
            "seed_scores": [
                evaluate_group(q2_truth, prediction, CAPACITY).score
                for prediction in q2_seed
            ],
        },
        "locked_h2": {
            "vs_current": h2_comparison,
            "vs_catboost_parent": parent_comparison,
            "correlation_with_catboost_parent": parent_correlation,
            "correlation_with_current": float(
                np.corrcoef(h2_current, h2_prediction)[0, 1]
            ),
            "seed_deltas_vs_current": seed_deltas,
            "monthly_deltas_vs_current": monthly,
            "positive_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "day_bootstrap": bootstrap,
        },
        "movement": {
            "changed_ratio": float((np.abs(movement) > 1e-9).mean()),
            "mean_absolute_kwh": float(np.abs(movement).mean()),
            "p95_absolute_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "max_absolute_kwh": float(np.abs(movement).max()),
        },
        "runtime_seconds": duration,
        "decision": (
            "Diversity diagnostic only; a Q2-selected bounded ensemble child is "
            "required before test prediction or submission generation."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "group3_xgboost_diversity_cache.npz"
    np.savez_compressed(
        cache_path,
        q2_index_ns=q2_index.astype("int64").to_numpy(),
        q2_truth=q2_truth.astype("float32"),
        q2_current=q2_current.astype("float32"),
        q2_seed_predictions=np.column_stack(q2_seed).astype("float32"),
        h2_index_ns=h2_index.astype("int64").to_numpy(),
        h2_truth=h2_truth.astype("float32"),
        h2_current=h2_current.astype("float32"),
        h2_seed_predictions=np.column_stack(h2_seed).astype("float32"),
        h2_prediction=h2_prediction.astype("float32"),
    )
    report_path = artifact_dir / "group3_xgboost_diversity_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    delta = h2_comparison["delta"]
    evaluation = {
        "family": "group3_xgboost_diversity",
        "locked_score_delta": delta["score"],
        "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
        "locked_ficr_delta": delta["ficr"],
        "expected_macro_score_delta": delta["score"] / 3.0,
        "positive_months": report["locked_h2"]["positive_months"],
        "total_months": 6,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "bootstrap_q05": bootstrap["q05"],
        "changed_ratio": report["movement"]["changed_ratio"],
        "p95_movement_ratio": report["movement"]["p95_absolute_kwh"] / CAPACITY,
        "fold_scores": [row["delta"]["score"] for row in seed_deltas],
        "cv_mean": float(np.mean([row["delta"]["score"] for row in seed_deltas])),
        "cv_std": float(np.std([row["delta"]["score"] for row in seed_deltas])),
        "oof_path": str(cache_path),
        "runtime_seconds": duration,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": delta["score"] / 3.0,
        "selection_direction": "maximize",
        "notes": "Full replacement diagnostic; no submission candidate is produced.",
    }
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache", default="artifacts_final/feature_cache/features_train.pkl"
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--parent-cache",
        default=(
            "artifacts_final/agent_service/runs/1/physical_catboost/"
            "physical_catboost_cache.npz"
        ),
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--agent-evaluation-output", required=True)
    parser.add_argument("--seeds", default="95001,95002,95003")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = run_experiment(
        Path(args.feature_cache),
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.parent_cache),
        Path(args.artifact_dir),
        Path(args.agent_evaluation_output),
        seeds,
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
