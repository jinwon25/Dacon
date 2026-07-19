from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.blocked_rolling_validation import (
    assign_issue_blocks,
    issue_block_bootstrap,
    load_issue_times,
)
from experiments.group3_physical_catboost import _compare
from src.metrics import CAPACITY_KWH
from train import make_model, select_feature_columns


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
TRAIN_START = pd.Timestamp("2023-01-01 01:00:00")
Q2_START = pd.Timestamp("2024-04-01 01:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
SEEDS = (84101, 84102, 84103)
MIN_WEIGHT = 0.25
MAX_WEIGHT = 4.0
MIN_LOCKED_SCORE_DELTA = 0.00015
MIN_BOOTSTRAP_POSITIVE_FRACTION = 0.80


@dataclass(frozen=True)
class BlendPolicy:
    alpha: float
    min_disagreement: float
    max_disagreement: float
    max_uncertainty: float


def weather_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"hour", "month", "dayofweek", "lead_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    return [column for column in select_feature_columns(features, TARGET, "own_idw") if column not in excluded]


def bounded_mean_one_weights(
    raw: np.ndarray,
    lower: float = MIN_WEIGHT,
    upper: float = MAX_WEIGHT,
) -> np.ndarray:
    """Project positive ratios to bounded, mean-one importance weights.

    A plain ``clip`` followed by division by the clipped mean does not preserve
    the requested bounds.  We instead solve for the multiplicative scale inside
    the clipping operation.  The solution is monotone and therefore a stable
    bisection is sufficient.
    """
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("raw weights must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("raw weights must be finite and strictly positive")
    if not (0.0 < lower <= 1.0 <= upper):
        raise ValueError("weight bounds must satisfy 0 < lower <= 1 <= upper")

    low_scale, high_scale = 0.0, 1.0
    while float(np.clip(values * high_scale, lower, upper).mean()) < 1.0:
        high_scale *= 2.0
    for _ in range(80):
        scale = 0.5 * (low_scale + high_scale)
        if float(np.clip(values * scale, lower, upper).mean()) < 1.0:
            low_scale = scale
        else:
            high_scale = scale
    weights = np.clip(values * high_scale, lower, upper)
    if not np.isclose(weights.mean(), 1.0, atol=1e-12):
        raise RuntimeError("bounded weight normalization failed")
    return weights


def covariate_weights(
    source: pd.DataFrame,
    target: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate clipped density-ratio weights with OOF domain probabilities."""
    combined = pd.concat([source, target], axis=0, ignore_index=True)
    labels = np.r_[np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
    probabilities = np.zeros(len(combined), dtype=float)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_rows, valid_rows) in enumerate(splitter.split(combined, labels)):
        model = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=24,
            min_child_samples=80,
            colsample_bytree=0.40,
            reg_lambda=2.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(combined.iloc[train_rows], labels[train_rows], callbacks=[lgb.log_evaluation(0)])
        probabilities[valid_rows] = model.predict_proba(combined.iloc[valid_rows])[:, 1]
    source_probability = np.clip(probabilities[: len(source)], 0.02, 0.98)
    prior_correction = len(source) / max(len(target), 1)
    raw = source_probability / (1.0 - source_probability) * prior_correction
    raw /= max(float(np.mean(raw)), 1e-9)
    weights = bounded_mean_one_weights(raw)
    diagnostics = {
        "source_probability_mean": float(source_probability.mean()),
        "target_probability_mean": float(probabilities[len(source) :].mean()),
        "weight_min": float(weights.min()),
        "weight_p50": float(np.quantile(weights, 0.50)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "clipped_fraction": float(np.mean((weights <= MIN_WEIGHT + 1e-12) | (weights >= MAX_WEIGHT - 1e-12))),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
    return weights, diagnostics


def fit_selection_members(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    domain_columns: list[str],
) -> tuple[np.ndarray, list[int], list[np.ndarray], dict[str, object]]:
    weights, domain = covariate_weights(X.loc[train_mask, domain_columns], X.loc[valid_mask, domain_columns], SEEDS[0])
    members, iterations = [], []
    for seed in SEEDS:
        model = make_model(seed)
        model.fit(
            X.loc[train_mask],
            y.loc[train_mask],
            sample_weight=weights,
            eval_set=[(X.loc[valid_mask], y.loc[valid_mask])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        members.append(np.clip(model.predict(X.loc[valid_mask]), 0.0, CAPACITY))
        iterations.append(max(100, int(model.best_iteration_)))
    return np.mean(members, axis=0), iterations, members, domain


def fit_locked_members(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    domain_columns: list[str],
    iterations: list[int],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, object]]:
    weights, domain = covariate_weights(X.loc[train_mask, domain_columns], X.loc[valid_mask, domain_columns], SEEDS[0] + 100)
    members = []
    for seed, n_estimators in zip(SEEDS, iterations, strict=True):
        model = make_model(seed, n_estimators)
        model.fit(X.loc[train_mask], y.loc[train_mask], sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
        members.append(np.clip(model.predict(X.loc[valid_mask]), 0.0, CAPACITY))
    return np.mean(members, axis=0), members, domain


def apply_policy(current: np.ndarray, members: np.ndarray, policy: BlendPolicy) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    members = np.asarray(members, dtype=float)
    mean = members.mean(axis=1)
    delta = mean - current
    disagreement = np.abs(delta) / CAPACITY
    uncertainty = members.std(axis=1) / CAPACITY
    directions = members - current[:, None]
    unanimous = np.all(directions >= 0.0, axis=1) | np.all(directions <= 0.0, axis=1)
    gate = (
        (current >= 0.10 * CAPACITY)
        & unanimous
        & (disagreement >= policy.min_disagreement)
        & (disagreement <= policy.max_disagreement)
        & (uncertainty <= policy.max_uncertainty)
    )
    candidate = current.copy()
    candidate[gate] = np.clip(current[gate] + policy.alpha * delta[gate], 0.0, CAPACITY)
    return candidate, gate


def select_policy(
    truth: np.ndarray,
    current: np.ndarray,
    members: np.ndarray,
    month_blocks: np.ndarray,
) -> tuple[BlendPolicy, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    uncertainty = members.std(axis=1) / CAPACITY
    std_values = sorted({0.0025, 0.005, 0.01, float(np.quantile(uncertainty, 0.25)), float(np.quantile(uncertainty, 0.50))})
    month_blocks = np.asarray(month_blocks, dtype=str)
    if len(month_blocks) != len(truth):
        raise ValueError("month_blocks and truth must have the same length")
    for alpha in (0.02, 0.05, 0.10, 0.15, 0.20, 0.225, 0.25):
        for minimum in (0.0, 0.0025, 0.005, 0.01):
            for maximum in (0.01, 0.02, 0.04, 0.06):
                if minimum >= maximum:
                    continue
                for max_std in std_values:
                    policy = BlendPolicy(alpha, minimum, maximum, max_std)
                    candidate, gate = apply_policy(current, members, policy)
                    coverage = float(gate.mean())
                    if coverage < 0.01 or coverage > 0.20:
                        continue
                    result = _compare(truth, current, candidate)
                    delta = result["delta"]
                    monthly = {}
                    for month in sorted(set(month_blocks)):
                        mask = month_blocks == month
                        monthly[month] = float(
                            _compare(truth[mask], current[mask], candidate[mask])["delta"]["score"]
                        )
                    if (
                        delta["one_minus_nmae"] < 0.0
                        or delta["ficr"] < 0.0
                        or sum(value > 0.0 for value in monthly.values()) < 2
                    ):
                        continue
                    rows.append({"policy": asdict(policy), "coverage": coverage, "metrics": result, "monthly": monthly})
    if not rows:
        raise RuntimeError("No covariate-shift blend passed development constraints")
    rows.sort(key=lambda row: (row["metrics"]["delta"]["score"], -row["coverage"]), reverse=True)
    return BlendPolicy(**rows[0]["policy"]), rows[:20]


def assess_promotion(
    locked: dict[str, dict[str, float]],
    monthly: dict[str, dict[str, float]],
    bootstrap: dict[str, float | int],
    changed_ratio: float,
    p95_movement_ratio: float,
) -> dict[str, object]:
    """Apply the deterministic BARAM promotion subset available here."""
    delta = locked["delta"]
    checks = {
        "locked_score_delta": delta["score"] >= MIN_LOCKED_SCORE_DELTA,
        "locked_one_minus_nmae_delta": delta["one_minus_nmae"] >= 0.0,
        "locked_ficr_delta": delta["ficr"] >= 0.0,
        "positive_month_fraction": (
            sum(row["score"] > 0.0 for row in monthly.values()) / max(len(monthly), 1)
        )
        >= 0.50,
        "bootstrap_positive_fraction": float(bootstrap["positive_fraction"])
        >= MIN_BOOTSTRAP_POSITIVE_FRACTION,
        "bootstrap_q05": float(bootstrap["q05"]) >= -0.00025,
        "changed_ratio": changed_ratio <= 0.25,
        "p95_movement_ratio": p95_movement_ratio <= 0.015,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_gates": failed}


def run(
    feature_path: Path,
    labels_path: Path,
    driver_path: Path,
    meta_path: Path,
    issue_source_path: Path,
    artifact_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    features = pd.read_pickle(feature_path)
    columns = select_feature_columns(features, TARGET, "own_idw")
    X = features[columns]
    domain_columns = [column for column in weather_columns(features) if column in X]
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    y = labels.set_index("kst_dtm").reindex(features.index)[TARGET]
    driver = np.load(driver_path)
    meta = np.load(meta_path)
    valid_index = pd.DatetimeIndex(pd.to_datetime(driver[f"{TARGET}__valid_index_ns"]))
    current = pd.Series(meta["valid_candidate"].astype(float), index=pd.to_datetime(meta["valid_index_ns"]))
    truth = pd.Series(driver[f"{TARGET}__valid_truth"].astype(float), index=valid_index)
    issue_times = load_issue_times(issue_source_path, features.index)

    q2_train = (features.index >= TRAIN_START) & (features.index < Q2_START) & y.notna() & (y >= 0.10 * CAPACITY)
    q2 = (features.index >= Q2_START) & (features.index < H2_START) & y.notna()
    h2_train = (features.index >= TRAIN_START) & (features.index < H2_START) & y.notna() & (y >= 0.10 * CAPACITY)
    h2 = (features.index >= H2_START) & y.notna()

    _, iterations, q2_members_list, q2_domain = fit_selection_members(X, y, np.asarray(q2_train), np.asarray(q2), domain_columns)
    q2_index = features.index[q2]
    q2_truth = y.loc[q2].to_numpy(dtype=float)
    q2_current = current.reindex(q2_index).to_numpy(dtype=float)
    q2_members = np.column_stack(q2_members_list)
    q2_issue = issue_times[q2]
    q2_months, _ = assign_issue_blocks(q2_index, q2_issue)
    policy, leaderboard = select_policy(q2_truth, q2_current, q2_members, q2_months)

    _, h2_members_list, h2_domain = fit_locked_members(X, y, np.asarray(h2_train), np.asarray(h2), domain_columns, iterations)
    h2_index = features.index[h2]
    h2_truth = truth.reindex(h2_index).to_numpy(dtype=float)
    h2_current = current.reindex(h2_index).to_numpy(dtype=float)
    h2_members = np.column_stack(h2_members_list)
    h2_candidate, h2_gate = apply_policy(h2_current, h2_members, policy)
    locked = _compare(h2_truth, h2_current, h2_candidate)
    h2_issue = issue_times[h2]
    h2_months, h2_seasons = assign_issue_blocks(h2_index, h2_issue)
    monthly = {}
    for month in sorted(set(h2_months)):
        mask = h2_months == month
        monthly[month] = _compare(h2_truth[mask], h2_current[mask], h2_candidate[mask])["delta"]
    bootstrap = issue_block_bootstrap(
        h2_truth,
        h2_current,
        h2_candidate,
        h2_issue,
        h2_seasons,
        np.ones(len(h2_index), dtype=bool),
        2000,
        seed=84_100,
    )
    changed_ratio = float(h2_gate.mean())
    p95_movement_ratio = float(np.quantile(np.abs(h2_candidate - h2_current) / CAPACITY, 0.95))
    promotion = assess_promotion(
        locked,
        monthly,
        bootstrap,
        changed_ratio,
        p95_movement_ratio,
    )
    report = {
        "method": "unlabeled-target covariate-shift importance weighted group3 LightGBM",
        "selection": {"policy": asdict(policy), "top": leaderboard, "domain": q2_domain, "iterations": iterations},
        "locked_h2": {
            "metrics": locked,
            "changed_rows": int(h2_gate.sum()),
            "changed_ratio": changed_ratio,
            "p95_movement_ratio": p95_movement_ratio,
            "monthly": monthly,
            "positive_months": int(sum(row["score"] > 0.0 for row in monthly.values())),
            "bootstrap": bootstrap,
            "domain": h2_domain,
        },
        "promotion": promotion,
        "candidate_created": False,
        "candidate_reason": (
            "diagnostic experiment; failed deterministic gates: "
            + ", ".join(promotion["failed_gates"])
            if promotion["failed_gates"]
            else "diagnostic experiment has no test-fit candidate path"
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "covariate_shift_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts_final/feature_cache/features_train.pkl")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--meta", default="artifacts_final/meta_gate/meta_gate_cache.npz")
    parser.add_argument("--issue-source", default="data/train/ldaps_train.csv")
    parser.add_argument("--artifact-dir", default="artifacts_final/covariate_shift")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                Path(args.features),
                Path(args.labels),
                Path(args.driver),
                Path(args.meta),
                Path(args.issue_source),
                Path(args.artifact_dir),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
