from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.group3_physical_catboost import (
    H2_START,
    Q2_START,
    _bootstrap_days,
    _compare,
)
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, select_feature_columns


TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITIES = np.asarray([CAPACITY_KWH[target] for target in TARGETS], dtype=float)
GROUP3 = TARGETS[2]
GROUP3_CAPACITY = CAPACITY_KWH[GROUP3]


def make_multitask_targets(labels: pd.DataFrame) -> np.ndarray:
    """Normalize outputs and mask rows excluded by the official group metric."""
    values = labels.loc[:, TARGETS].to_numpy(dtype=float)
    normalized = values / CAPACITIES
    eligible = values >= (0.10 * CAPACITIES)
    normalized[~eligible] = np.nan
    return normalized


def _model(seed: int, iterations: int = 1_200) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSEWithMissingValues",
        eval_metric="MultiRMSEWithMissingValues",
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


def _rows(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _fit_select(
    X: pd.DataFrame,
    targets: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    iterations: list[int] = []
    train_rows = _rows(train_mask)
    valid_rows = _rows(valid_mask)
    for seed in seeds:
        model = _model(seed)
        model.fit(
            X.iloc[train_rows],
            targets[train_rows],
            eval_set=(X.iloc[valid_rows], targets[valid_rows]),
            use_best_model=True,
        )
        prediction = np.asarray(model.predict(X.iloc[valid_rows]), dtype=float)
        predictions.append(prediction)
        iterations.append(max(100, int(model.get_best_iteration() + 1)))
    return np.mean(predictions, axis=0), iterations, predictions


def _fit_locked(
    X: pd.DataFrame,
    targets: np.ndarray,
    train_mask: np.ndarray,
    prediction_mask: np.ndarray,
    seeds: tuple[int, ...],
    iterations: list[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    train_rows = _rows(train_mask)
    prediction_rows = _rows(prediction_mask)
    for seed, n_iterations in zip(seeds, iterations, strict=True):
        model = _model(seed, n_iterations)
        model.fit(X.iloc[train_rows], targets[train_rows])
        predictions.append(
            np.asarray(model.predict(X.iloc[prediction_rows]), dtype=float)
        )
    return np.mean(predictions, axis=0), predictions


def _current_predictions(
    driver: np.lib.npyio.NpzFile,
    meta: np.lib.npyio.NpzFile,
    index: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for target in TARGETS[:2]:
        driver_index = pd.DatetimeIndex(
            pd.to_datetime(driver[f"{target}__valid_index_ns"])
        )
        output[target] = (
            pd.Series(driver[f"{target}__exact_base"].astype(float), index=driver_index)
            .reindex(index)
            .to_numpy(dtype=float)
        )
    meta_index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    output[GROUP3] = (
        pd.Series(meta["valid_candidate"].astype(float), index=meta_index)
        .reindex(index)
        .to_numpy(dtype=float)
    )
    if not all(np.isfinite(values).all() for values in output.values()):
        raise ValueError("Current exact prediction lineage does not cover evaluation index")
    return output


def _truth_predictions(
    driver: np.lib.npyio.NpzFile, index: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for target in TARGETS:
        driver_index = pd.DatetimeIndex(
            pd.to_datetime(driver[f"{target}__valid_index_ns"])
        )
        output[target] = (
            pd.Series(driver[f"{target}__valid_truth"].astype(float), index=driver_index)
            .reindex(index)
            .to_numpy(dtype=float)
        )
    if not all(np.isfinite(values).all() for values in output.values()):
        raise ValueError("Exact truth lineage does not cover evaluation index")
    return output


def _calibrate_outputs(
    truth: dict[str, np.ndarray], normalized_prediction: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    output: dict[str, np.ndarray] = {}
    calibration: list[dict[str, float]] = []
    for column, target in enumerate(TARGETS):
        capacity = CAPACITY_KWH[target]
        raw_kwh = normalized_prediction[:, column] * capacity
        scale, offset, _ = calibrate(truth[target], raw_kwh, capacity)
        output[target] = np.clip(raw_kwh * scale + offset, 0.0, capacity)
        calibration.append({"scale": float(scale), "offset": float(offset)})
    return output, calibration


def _apply_calibration(
    normalized_prediction: np.ndarray,
    calibration: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for column, target in enumerate(TARGETS):
        capacity = CAPACITY_KWH[target]
        params = calibration[column]
        output[target] = np.clip(
            normalized_prediction[:, column] * capacity * params["scale"]
            + params["offset"],
            0.0,
            capacity,
        )
    return output


def _metric_delta(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, float]:
    return {
        "score": float(after["score"]) - float(before["score"]),
        "one_minus_nmae": float(after["one_minus_nmae"])
        - float(before["one_minus_nmae"]),
        "ficr": float(after["ficr"]) - float(before["ficr"]),
    }


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
    columns = select_feature_columns(features, GROUP3, "base")
    X = features[columns]
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(features.index)
    targets = make_multitask_targets(labels)
    at_least_one_target = np.isfinite(targets).any(axis=1)

    q2_train = (features.index < Q2_START) & at_least_one_target
    q2_eval = (
        (features.index >= Q2_START)
        & (features.index < H2_START)
        & at_least_one_target
    )
    h2_train = (features.index < H2_START) & at_least_one_target
    h2_predict = (features.index >= H2_START) & labels[GROUP3].notna().to_numpy()

    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    q2_index = features.index[q2_eval]
    h2_index = features.index[h2_predict]
    q2_truth = _truth_predictions(driver, q2_index)
    h2_truth = _truth_predictions(driver, h2_index)
    q2_current = _current_predictions(driver, meta, q2_index)
    h2_current = _current_predictions(driver, meta, h2_index)

    q2_mean, iterations, q2_seed_raw = _fit_select(
        X, targets, q2_train, q2_eval, seeds
    )
    q2_prediction, calibration = _calibrate_outputs(q2_truth, q2_mean)
    q2_before = evaluate_competition(q2_truth, q2_current)
    q2_after = evaluate_competition(q2_truth, q2_prediction)

    h2_mean, h2_seed_raw = _fit_locked(
        X, targets, h2_train, h2_predict, seeds, iterations
    )
    h2_prediction = _apply_calibration(h2_mean, calibration)
    h2_before = evaluate_competition(h2_truth, h2_current)
    h2_after = evaluate_competition(h2_truth, h2_prediction)

    group3_locked = _compare(
        h2_truth[GROUP3], h2_current[GROUP3], h2_prediction[GROUP3]
    )
    parent_comparison: dict[str, object] | None = None
    if parent_cache_path.exists():
        parent = np.load(parent_cache_path, allow_pickle=True)
        parent_index = pd.DatetimeIndex(pd.to_datetime(parent["h2_index_ns"]))
        parent_control = (
            pd.Series(parent["h2_base_control"].astype(float), index=parent_index)
            .reindex(h2_index)
            .to_numpy(dtype=float)
        )
        if np.isfinite(parent_control).all():
            parent_comparison = _compare(
                h2_truth[GROUP3], parent_control, h2_prediction[GROUP3]
            )

    seed_deltas: list[dict[str, object]] = []
    for seed, raw in zip(seeds, h2_seed_raw, strict=True):
        prediction = _apply_calibration(raw, calibration)
        seed_deltas.append(
            {
                "seed": seed,
                "group3_delta_vs_current": _compare(
                    h2_truth[GROUP3], h2_current[GROUP3], prediction[GROUP3]
                )["delta"],
                "competition_delta_vs_current": _metric_delta(
                    h2_before, evaluate_competition(h2_truth, prediction)
                ),
            }
        )

    monthly: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        mask = h2_index.month == month
        monthly[str(month)] = _compare(
            h2_truth[GROUP3][mask],
            h2_current[GROUP3][mask],
            h2_prediction[GROUP3][mask],
        )["delta"]
    bootstrap = _bootstrap_days(
        h2_truth[GROUP3],
        h2_current[GROUP3],
        h2_prediction[GROUP3],
        h2_index,
        n_bootstrap,
    )
    movement = h2_prediction[GROUP3] - h2_current[GROUP3]
    duration = time.perf_counter() - started
    q2_group3 = _compare(
        q2_truth[GROUP3], q2_current[GROUP3], q2_prediction[GROUP3]
    )

    report: dict[str, object] = {
        "method": "eligible-target masked multi-output CatBoost",
        "hypothesis": (
            "Shared multi-output trees can transfer the two auxiliary farms' 2022 labels "
            "to the short-history group-3 representation without fabricating group-3 labels."
        ),
        "n_features": len(columns),
        "seeds": list(seeds),
        "target_observations": {
            target: int(np.isfinite(targets[:, column]).sum())
            for column, target in enumerate(TARGETS)
        },
        "train_contract": {
            "selection": "all available eligible targets before 2024-Q2 -> 2024-Q2",
            "locked": "all available eligible targets before 2024-H2 -> 2024-H2",
            "group3_2022_labels": "missing, handled natively by objective",
            "target_scale": "capacity-normalized",
        },
        "selection_q2": {
            "best_iterations": iterations,
            "calibration": {
                target: calibration[column]
                for column, target in enumerate(TARGETS)
            },
            "competition_before": q2_before,
            "competition_after": q2_after,
            "competition_delta": _metric_delta(q2_before, q2_after),
            "group3_vs_current": q2_group3,
            "group3_seed_scores": [
                evaluate_group(
                    q2_truth[GROUP3],
                    _apply_calibration(raw, calibration)[GROUP3],
                    GROUP3_CAPACITY,
                ).score
                for raw in q2_seed_raw
            ],
        },
        "locked_h2": {
            "competition_before": h2_before,
            "competition_after": h2_after,
            "competition_delta": _metric_delta(h2_before, h2_after),
            "group3_vs_current": group3_locked,
            "group3_vs_parent_single_task_control": parent_comparison,
            "seed_deltas": seed_deltas,
            "monthly_deltas": monthly,
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
            "Structural diagnostic only. A bounded child is required before any test "
            "prediction or submission CSV can be created."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "group_multitask_catboost_cache.npz"
    np.savez_compressed(
        cache_path,
        q2_index_ns=q2_index.astype("int64").to_numpy(),
        q2_group3_truth=q2_truth[GROUP3].astype("float32"),
        q2_group3_current=q2_current[GROUP3].astype("float32"),
        q2_group3_multitask=q2_prediction[GROUP3].astype("float32"),
        h2_index_ns=h2_index.astype("int64").to_numpy(),
        h2_group3_truth=h2_truth[GROUP3].astype("float32"),
        h2_group3_current=h2_current[GROUP3].astype("float32"),
        h2_group3_multitask=h2_prediction[GROUP3].astype("float32"),
        h2_multitask_all=np.column_stack(
            [h2_prediction[target] for target in TARGETS]
        ).astype("float32"),
    )
    report_path = artifact_dir / "group_multitask_catboost_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    group3_delta = group3_locked["delta"]
    evaluation = {
        "family": "group_multitask_catboost",
        "locked_score_delta": group3_delta["score"],
        "locked_one_minus_nmae_delta": group3_delta["one_minus_nmae"],
        "locked_ficr_delta": group3_delta["ficr"],
        "expected_macro_score_delta": group3_delta["score"] / 3.0,
        "full_competition_score_delta": report["locked_h2"]["competition_delta"][
            "score"
        ],
        "positive_months": report["locked_h2"]["positive_months"],
        "total_months": 6,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "bootstrap_q05": bootstrap["q05"],
        "changed_ratio": report["movement"]["changed_ratio"],
        "p95_movement_ratio": report["movement"]["p95_absolute_kwh"]
        / GROUP3_CAPACITY,
        "fold_scores": [
            row["group3_delta_vs_current"]["score"] for row in seed_deltas
        ],
        "cv_mean": float(
            np.mean([row["group3_delta_vs_current"]["score"] for row in seed_deltas])
        ),
        "cv_std": float(
            np.std([row["group3_delta_vs_current"]["score"] for row in seed_deltas])
        ),
        "oof_path": str(cache_path),
        "runtime_seconds": duration,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": group3_delta["score"] / 3.0,
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
    parser.add_argument(
        "--parent-cache",
        default=(
            "artifacts_final/agent_service/runs/1/physical_catboost/"
            "physical_catboost_cache.npz"
        ),
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--agent-evaluation-output", required=True)
    parser.add_argument("--seeds", default="73001,73002,73003")
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
