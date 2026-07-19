from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.group3_physical_catboost import (
    CAPACITY,
    H2_START,
    Q2_START,
    TARGET,
    _bootstrap_days,
    _compare,
    _model,
)
from experiments.phase_regime_cross_group import lead_phase
from src.metrics import evaluate_group
from train import calibrate, select_feature_columns


def _fit_phase_select(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
    phases = lead_phase(X.index)
    valid_rows = np.flatnonzero(valid_mask)
    valid_phases = phases[valid_mask]
    seed_predictions: list[np.ndarray] = []
    selected_iterations: list[list[int]] = []
    for seed in seeds:
        prediction = np.full(valid_mask.sum(), np.nan, dtype=float)
        seed_iterations: list[int] = []
        for phase in range(4):
            phase_train = train_mask & (phases == phase)
            phase_valid = valid_mask & (phases == phase)
            if phase_train.sum() < 500 or phase_valid.sum() < 100:
                raise ValueError(f"Insufficient rows for lead phase {phase}")
            model = _model(seed + 100 * phase)
            model.fit(
                X.iloc[np.flatnonzero(phase_train)],
                y.iloc[np.flatnonzero(phase_train)],
                eval_set=(
                    X.iloc[np.flatnonzero(phase_valid)],
                    y.iloc[np.flatnonzero(phase_valid)],
                ),
                use_best_model=True,
            )
            phase_prediction = np.clip(
                model.predict(X.iloc[np.flatnonzero(phase_valid)]), 0.0, CAPACITY
            )
            prediction[valid_phases == phase] = phase_prediction
            seed_iterations.append(max(100, int(model.get_best_iteration() + 1)))
        if not np.isfinite(prediction).all():
            missing = valid_rows[~np.isfinite(prediction)]
            raise ValueError(f"Lead-phase prediction has {len(missing)} missing rows")
        seed_predictions.append(prediction)
        selected_iterations.append(seed_iterations)
    return np.mean(seed_predictions, axis=0), selected_iterations, seed_predictions


def _fit_phase_locked(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    seeds: tuple[int, ...],
    selected_iterations: list[list[int]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    phases = lead_phase(X.index)
    valid_phases = phases[valid_mask]
    seed_predictions: list[np.ndarray] = []
    for seed, phase_iterations in zip(seeds, selected_iterations, strict=True):
        prediction = np.full(valid_mask.sum(), np.nan, dtype=float)
        for phase in range(4):
            phase_train = train_mask & (phases == phase)
            phase_valid = valid_mask & (phases == phase)
            model = _model(seed + 100 * phase, phase_iterations[phase])
            model.fit(
                X.iloc[np.flatnonzero(phase_train)],
                y.iloc[np.flatnonzero(phase_train)],
            )
            phase_prediction = np.clip(
                model.predict(X.iloc[np.flatnonzero(phase_valid)]), 0.0, CAPACITY
            )
            prediction[valid_phases == phase] = phase_prediction
        if not np.isfinite(prediction).all():
            raise ValueError("Locked lead-phase prediction contains missing rows")
        seed_predictions.append(prediction)
    return np.mean(seed_predictions, axis=0), seed_predictions


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
        raise ValueError("Exact current lineage does not cover requested horizon index")
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

    q2_mean, iterations, q2_seed_raw = _fit_phase_select(
        X, y, q2_train, q2_valid, seeds
    )
    scale, offset, _ = calibrate(q2_truth, q2_mean, CAPACITY)
    q2_seed = [
        np.clip(prediction * scale + offset, 0.0, CAPACITY)
        for prediction in q2_seed_raw
    ]
    q2_prediction = np.clip(q2_mean * scale + offset, 0.0, CAPACITY)

    h2_mean, h2_seed_raw = _fit_phase_locked(
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
    phase_deltas = {
        str(phase): _compare(
            h2_truth[lead_phase(h2_index) == phase],
            h2_current[lead_phase(h2_index) == phase],
            h2_prediction[lead_phase(h2_index) == phase],
        )["delta"]
        for phase in range(4)
    }
    bootstrap = _bootstrap_days(
        h2_truth, h2_current, h2_prediction, h2_index, n_bootstrap
    )
    movement = h2_prediction - h2_current
    duration = time.perf_counter() - started

    report: dict[str, object] = {
        "method": "four direct CatBoost experts partitioned by six-hour NWP lead phase",
        "hypothesis": (
            "Direct group-3 learners partitioned by forecast lead phase can capture "
            "horizon-specific NWP bias that a monolithic CatBoost only sees as a feature."
        ),
        "n_features": len(columns),
        "seeds": list(seeds),
        "train_contract": {
            "selection": "2023-01-01 through 2024-Q1 -> 2024-Q2",
            "locked": "2023-01-01 through 2024-H1 -> 2024-H2",
            "lead_phases": ["12-17", "18-23", "24-29", "30-35"],
            "eligible_training_only": True,
        },
        "selection_q2": {
            "best_iterations_by_seed_phase": iterations,
            "calibration": {"scale": scale, "offset": offset},
            "vs_current": q2_comparison,
            "seed_scores": [
                evaluate_group(q2_truth, prediction, CAPACITY).score
                for prediction in q2_seed
            ],
        },
        "locked_h2": {
            "vs_current": h2_comparison,
            "vs_monolithic_parent": parent_comparison,
            "seed_deltas_vs_current": seed_deltas,
            "monthly_deltas_vs_current": monthly,
            "positive_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "lead_phase_deltas_vs_current": phase_deltas,
            "positive_lead_phases": int(
                sum(value["score"] > 0.0 for value in phase_deltas.values())
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
            "Structural diagnostic only; a separately validated bounded child is "
            "required before test prediction or submission generation."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "group3_horizon_catboost_cache.npz"
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
    report_path = artifact_dir / "group3_horizon_catboost_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    delta = h2_comparison["delta"]
    evaluation = {
        "family": "group3_horizon_catboost",
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
    parser.add_argument("--seeds", default="84001,84002,84003")
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
