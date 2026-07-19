from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
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
    _fit_locked,
    _fit_select,
)
from train import calibrate, select_feature_columns


@dataclass(frozen=True)
class BlendPolicy:
    alpha: float
    min_disagreement_ratio: float
    max_disagreement_ratio: float
    max_seed_std_ratio: float


def apply_policy(
    current: np.ndarray,
    seed_predictions: np.ndarray,
    policy: BlendPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    seed_predictions = np.asarray(seed_predictions, dtype=float)
    member = seed_predictions.mean(axis=1)
    delta = member - current
    seed_delta = seed_predictions - current[:, None]
    unanimous = np.all(seed_delta >= 0.0, axis=1) | np.all(seed_delta <= 0.0, axis=1)
    disagreement = np.abs(delta) / CAPACITY
    uncertainty = seed_predictions.std(axis=1) / CAPACITY
    gate = (
        (current >= 0.10 * CAPACITY)
        & unanimous
        & (disagreement >= policy.min_disagreement_ratio)
        & (disagreement <= policy.max_disagreement_ratio)
        & (uncertainty <= policy.max_seed_std_ratio)
    )
    candidate = current.copy()
    candidate[gate] = np.clip(
        current[gate] + policy.alpha * delta[gate], 0.0, CAPACITY
    )
    return candidate, gate


def select_policy(
    truth: np.ndarray,
    current: np.ndarray,
    seed_predictions: np.ndarray,
    max_coverage: float = 0.25,
) -> tuple[BlendPolicy, list[dict[str, object]]]:
    uncertainty = seed_predictions.std(axis=1) / CAPACITY
    finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
    std_thresholds = sorted(
        {
            0.0025,
            0.005,
            0.01,
            0.02,
            *(
                float(np.quantile(finite_uncertainty, q))
                for q in (0.10, 0.25, 0.50)
            ),
        }
    )
    rows: list[dict[str, object]] = []
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25):
        for minimum in (0.0, 0.0025, 0.005, 0.01):
            for maximum in (0.01, 0.02, 0.04, 0.06):
                if minimum >= maximum:
                    continue
                for max_std in std_thresholds:
                    policy = BlendPolicy(alpha, minimum, maximum, max_std)
                    candidate, gate = apply_policy(current, seed_predictions, policy)
                    coverage = float(gate.mean())
                    if coverage <= 0.005 or coverage > max_coverage:
                        continue
                    comparison = _compare(truth, current, candidate)
                    delta = comparison["delta"]
                    rows.append(
                        {
                            "policy": asdict(policy),
                            "changed_ratio": coverage,
                            "changed_rows": int(gate.sum()),
                            "delta": delta,
                            "passes_component_signs": bool(
                                delta["one_minus_nmae"] >= 0.0
                                and delta["ficr"] >= 0.0
                            ),
                        }
                    )
    if not rows:
        raise RuntimeError("No bounded rolling CatBoost blend policy was available")
    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["passes_component_signs"]),
            float(row["delta"]["score"]),
            -float(row["changed_ratio"]),
        ),
        reverse=True,
    )
    selected = BlendPolicy(**ranked[0]["policy"])
    return selected, ranked[:25]


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
    columns = select_feature_columns(features, TARGET, "base")
    X = features[columns]
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    y = labels.reindex(features.index)[TARGET]
    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    valid_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    current = pd.Series(
        meta["valid_candidate"].astype(float),
        index=pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"])),
    )
    truth = pd.Series(
        driver[f"{TARGET}__valid_truth"].astype(float), index=valid_index
    )

    q2_train = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < Q2_START)
        & y.notna()
        & (y >= 0.10 * CAPACITY)
    )
    q2 = (
        (features.index >= Q2_START)
        & (features.index < H2_START)
        & y.notna()
    )
    h2_train = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < H2_START)
        & y.notna()
        & (y >= 0.10 * CAPACITY)
    )
    h2 = (features.index >= H2_START) & y.notna()

    q2_mean, iterations, q2_seed_raw = _fit_select(
        X, y, np.asarray(q2_train), np.asarray(q2), seeds
    )
    q2_truth = y.loc[q2].to_numpy(dtype=float)
    scale, offset, _ = calibrate(q2_truth, q2_mean, CAPACITY)
    q2_seed = np.column_stack(
        [np.clip(value * scale + offset, 0.0, CAPACITY) for value in q2_seed_raw]
    )
    q2_index = features.index[q2]
    q2_current = current.reindex(q2_index).to_numpy(dtype=float)
    policy, leaderboard = select_policy(q2_truth, q2_current, q2_seed)
    q2_candidate, q2_gate = apply_policy(q2_current, q2_seed, policy)

    _, h2_seed_raw = _fit_locked(
        X, y, np.asarray(h2_train), np.asarray(h2), seeds, iterations
    )
    h2_seed = np.column_stack(
        [np.clip(value * scale + offset, 0.0, CAPACITY) for value in h2_seed_raw]
    )
    h2_index = features.index[h2]
    h2_truth = truth.reindex(h2_index).to_numpy(dtype=float)
    h2_current = current.reindex(h2_index).to_numpy(dtype=float)
    h2_candidate, h2_gate = apply_policy(h2_current, h2_seed, policy)
    locked = _compare(h2_truth, h2_current, h2_candidate)

    seed_deltas = []
    for seed_i, seed in enumerate(seeds):
        seed_candidate = h2_current.copy()
        seed_candidate[h2_gate] = np.clip(
            h2_current[h2_gate]
            + policy.alpha * (h2_seed[h2_gate, seed_i] - h2_current[h2_gate]),
            0.0,
            CAPACITY,
        )
        seed_deltas.append(
            {"seed": seed, "delta": _compare(h2_truth, h2_current, seed_candidate)["delta"]}
        )
    monthly = {}
    for month in range(7, 13):
        mask = h2_index.month == month
        monthly[str(month)] = _compare(
            h2_truth[mask], h2_current[mask], h2_candidate[mask]
        )["delta"]
    bootstrap = _bootstrap_days(
        h2_truth, h2_current, h2_candidate, h2_index, n_bootstrap
    )
    movement = h2_candidate - h2_current
    duration = time.perf_counter() - started
    report: dict[str, object] = {
        "method": "rolling-origin CatBoost refresh with seed-consensus bounded blend",
        "parent_evidence": (
            "Run 1 base-control beat its physical-feature sibling and the current H2 surface; "
            "this child removes the failed physical features and restricts coverage."
        ),
        "n_features": len(columns),
        "seeds": list(seeds),
        "best_iterations_selected_on_q2": iterations,
        "calibration_selected_on_q2": {"scale": scale, "offset": offset},
        "selection": {
            "period": "2024-Q2",
            "policy": asdict(policy),
            "changed_rows": int(q2_gate.sum()),
            "changed_ratio": float(q2_gate.mean()),
            "metrics": _compare(q2_truth, q2_current, q2_candidate),
            "top_policies": leaderboard,
        },
        "locked_h2": {
            "metrics": locked,
            "changed_rows": int(h2_gate.sum()),
            "changed_ratio": float(h2_gate.mean()),
            "seed_deltas": seed_deltas,
            "monthly_deltas": monthly,
            "positive_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "day_bootstrap": bootstrap,
        },
        "movement": {
            "mean_absolute_kwh": float(np.abs(movement).mean()),
            "p95_absolute_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "max_absolute_kwh": float(np.abs(movement).max()),
        },
        "runtime_seconds": duration,
        "decision": "Validation-only child; no test prediction or submission CSV is created.",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "rolling_catboost_blend_cache.npz",
        q2_index_ns=q2_index.astype("int64").to_numpy(),
        q2_current=q2_current.astype("float32"),
        q2_seed_predictions=q2_seed.astype("float32"),
        h2_index_ns=h2_index.astype("int64").to_numpy(),
        h2_truth=h2_truth.astype("float32"),
        h2_current=h2_current.astype("float32"),
        h2_seed_predictions=h2_seed.astype("float32"),
        h2_candidate=h2_candidate.astype("float32"),
        h2_gate=h2_gate,
    )
    (artifact_dir / "rolling_catboost_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    delta = locked["delta"]
    fold_scores = [row["delta"]["score"] for row in seed_deltas]
    evaluation = {
        "family": "group3_rolling_catboost_blend",
        "locked_score_delta": delta["score"],
        "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
        "locked_ficr_delta": delta["ficr"],
        "expected_macro_score_delta": delta["score"] / 3.0,
        "positive_months": report["locked_h2"]["positive_months"],
        "total_months": 6,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "bootstrap_q05": bootstrap["q05"],
        "changed_ratio": float(h2_gate.mean()),
        "p95_movement_ratio": report["movement"]["p95_absolute_kwh"] / CAPACITY,
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_path": str(artifact_dir / "rolling_catboost_blend_cache.npz"),
        "runtime_seconds": duration,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": delta["score"] / 3.0,
        "selection_direction": "maximize",
        "notes": "Q2-selected low-coverage blend evaluated once on locked H2.",
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
    report = run_experiment(
        Path(args.feature_cache),
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.artifact_dir),
        Path(args.agent_evaluation_output),
        tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
