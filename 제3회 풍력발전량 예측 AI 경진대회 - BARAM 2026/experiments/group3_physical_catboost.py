from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group
from train import calibrate, select_feature_columns


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def _model(seed: int, iterations: int = 1_200) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=120,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
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
    seed_predictions: list[np.ndarray] = []
    for seed in seeds:
        model = _model(seed)
        model.fit(
            X.loc[train_mask],
            y.loc[train_mask],
            eval_set=(X.loc[valid_mask], y.loc[valid_mask]),
            use_best_model=True,
        )
        prediction = np.clip(model.predict(X.loc[valid_mask]), 0.0, CAPACITY)
        predictions.append(prediction)
        seed_predictions.append(prediction)
        iterations.append(max(100, int(model.get_best_iteration() + 1)))
    return np.mean(predictions, axis=0), iterations, seed_predictions


def _fit_locked(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
    iterations: list[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    for seed, n_iterations in zip(seeds, iterations, strict=True):
        model = _model(seed, iterations=n_iterations)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        predictions.append(
            np.clip(model.predict(X.loc[valid_mask]), 0.0, CAPACITY)
        )
    return np.mean(predictions, axis=0), predictions


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _compare(
    truth: np.ndarray, before: np.ndarray, after: np.ndarray
) -> dict[str, object]:
    before_metric = evaluate_group(truth, before, CAPACITY)
    after_metric = evaluate_group(truth, after, CAPACITY)
    return {
        "before": before_metric.to_dict(),
        "after": after_metric.to_dict(),
        "delta": _metric_delta(before_metric, after_metric),
    }


def _bootstrap_days(
    truth: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    index: pd.DatetimeIndex,
    n_bootstrap: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(20260717)
    days = index.normalize().unique()
    positions = {
        day: np.flatnonzero(index.normalize() == day) for day in days
    }
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        deltas.append(
            evaluate_group(truth[rows], after[rows], CAPACITY).score
            - evaluate_group(truth[rows], before[rows], CAPACITY).score
        )
    values = np.asarray(deltas)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((values > 0.0).mean()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def run_experiment(
    feature_cache: Path,
    labels_path: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
    artifact_dir: Path,
    evaluation_output: Path,
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    features = pd.read_pickle(feature_cache)
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    y = labels.reindex(features.index)[TARGET]

    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    valid_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not np.array_equal(
        meta["valid_index_ns"], valid_index.astype("int64").to_numpy()
    ):
        raise ValueError("Meta and driver validation indexes differ")
    current = meta["valid_candidate"].astype(float)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)

    periods = {
        "q2_train": (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < Q2_START)
        & y.notna()
        & (y >= 0.10 * CAPACITY),
        "q2": (features.index >= Q2_START)
        & (features.index < H2_START)
        & y.notna(),
        "h2_train": (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < H2_START)
        & y.notna()
        & (y >= 0.10 * CAPACITY),
        "h2": (features.index >= H2_START) & y.notna(),
    }
    columns_by_name = {
        "base_control": select_feature_columns(features, TARGET, "base"),
        "physical_idw_hub": select_feature_columns(features, TARGET, "own_idw"),
    }
    results: dict[str, object] = {}
    locked_predictions: dict[str, np.ndarray] = {}
    locked_seed_predictions: dict[str, list[np.ndarray]] = {}

    for name, columns in columns_by_name.items():
        X = features[columns]
        q2_prediction, iterations, q2_seed_predictions = _fit_select(
            X,
            y,
            np.asarray(periods["q2_train"]),
            np.asarray(periods["q2"]),
            seeds,
        )
        q2_truth = y.loc[periods["q2"]].to_numpy(dtype=float)
        scale, offset, q2_metric = calibrate(q2_truth, q2_prediction, CAPACITY)
        q2_calibrated = np.clip(q2_prediction * scale + offset, 0.0, CAPACITY)
        h2_prediction, h2_seed_predictions = _fit_locked(
            X,
            y,
            np.asarray(periods["h2_train"]),
            np.asarray(periods["h2"]),
            seeds,
            iterations,
        )
        h2_calibrated = np.clip(h2_prediction * scale + offset, 0.0, CAPACITY)
        locked_predictions[name] = h2_calibrated
        locked_seed_predictions[name] = [
            np.clip(prediction * scale + offset, 0.0, CAPACITY)
            for prediction in h2_seed_predictions
        ]
        h2_truth = y.loc[periods["h2"]].to_numpy(dtype=float)
        results[name] = {
            "n_features": len(columns),
            "best_iterations": iterations,
            "calibration": {"scale": scale, "offset": offset},
            "q2": evaluate_group(q2_truth, q2_calibrated, CAPACITY).to_dict(),
            "q2_seed_scores": [
                evaluate_group(
                    q2_truth,
                    np.clip(prediction * scale + offset, 0.0, CAPACITY),
                    CAPACITY,
                ).score
                for prediction in q2_seed_predictions
            ],
            "h2": evaluate_group(h2_truth, h2_calibrated, CAPACITY).to_dict(),
        }

    h2_index = features.index[periods["h2"]]
    current_h2 = pd.Series(current, index=valid_index).reindex(h2_index).to_numpy()
    truth_h2 = pd.Series(truth, index=valid_index).reindex(h2_index).to_numpy()
    physical = locked_predictions["physical_idw_hub"]
    control = locked_predictions["base_control"]
    physical_vs_current = _compare(truth_h2, current_h2, physical)
    physical_vs_control = _compare(truth_h2, control, physical)

    monthly: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        mask = h2_index.month == month
        monthly[str(month)] = _compare(
            truth_h2[mask], current_h2[mask], physical[mask]
        )["delta"]
    bootstrap = _bootstrap_days(
        truth_h2, current_h2, physical, h2_index, n_bootstrap
    )
    seed_deltas = []
    for seed, prediction in zip(
        seeds, locked_seed_predictions["physical_idw_hub"], strict=True
    ):
        seed_deltas.append(
            {
                "seed": seed,
                "delta": _compare(truth_h2, current_h2, prediction)["delta"],
            }
        )
    movement = physical - current_h2
    duration = time.perf_counter() - started

    report: dict[str, object] = {
        "method": "expanding-window group-3 CatBoost physical feature surface",
        "hypothesis": (
            "Adding only group-3 turbine-location IDW and hub-height features to the "
            "same CatBoost learner improves a base-control model and the exact current surface."
        ),
        "seeds": list(seeds),
        "train_contract": {
            "q2_selection": "2023-01-01 through 2024-Q1 -> 2024-Q2",
            "locked": "2023-01-01 through 2024-H1 -> 2024-H2",
            "eligible_training_only": True,
        },
        "models": results,
        "locked_h2": {
            "physical_vs_current": physical_vs_current,
            "physical_vs_base_control": physical_vs_control,
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
            "Diagnostic member only. A child run may create a low-coverage ensemble only if "
            "locked score/FICR, seed direction, month stability, and bootstrap all support it."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "physical_catboost_cache.npz",
        h2_index_ns=h2_index.astype("int64").to_numpy(),
        h2_truth=truth_h2.astype("float32"),
        h2_current=current_h2.astype("float32"),
        h2_base_control=control.astype("float32"),
        h2_physical=physical.astype("float32"),
    )
    report_path = artifact_dir / "physical_catboost_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    delta = physical_vs_current["delta"]
    evaluation = {
        "family": "group3_physical_catboost",
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
        "oof_path": str(artifact_dir / "physical_catboost_cache.npz"),
        "runtime_seconds": duration,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": delta["score"] / 3.0,
        "selection_direction": "maximize",
        "notes": "Full replacement diagnostic; no submission candidate is produced by this run.",
    }
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--agent-evaluation-output", required=True)
    parser.add_argument("--seeds", default="62001,62002,62003")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = run_experiment(
        Path(args.feature_cache),
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.artifact_dir),
        Path(args.agent_evaluation_output),
        seeds,
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
