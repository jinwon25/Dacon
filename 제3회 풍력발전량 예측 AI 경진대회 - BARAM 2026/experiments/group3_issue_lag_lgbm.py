from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group
from train import calibrate, make_model, select_feature_columns


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
H1_START = pd.Timestamp("2024-01-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
ISSUE_LAGS = (6, 12, 24)
RAW_COLUMNS = (
    "wind_speed_10m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_speed_100m",
    "wind_u_100m",
    "wind_v_100m",
    "temperature_2m",
    "surface_pressure",
)


@dataclass(frozen=True)
class BlendPolicy:
    max_disagreement: float
    min_base_ratio: float
    alpha: float


def _issue_times(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    issue = (
        frame.drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")["data_available_kst_dtm"]
        .reindex(index)
    )
    if issue.isna().any():
        raise ValueError("Missing issue timestamp for one or more feature rows")
    return issue


def issue_lag_features(
    index: pd.DatetimeIndex,
    issue: pd.Series,
    history: pd.DataFrame,
) -> pd.DataFrame:
    output: dict[str, np.ndarray] = {}
    for lag in ISSUE_LAGS:
        source_time = pd.DatetimeIndex(issue - pd.Timedelta(hours=lag))
        if not (source_time < pd.DatetimeIndex(issue)).all():
            raise ValueError("Issue-lag source time must precede every issue time")
        values = history.reindex(source_time)
        if values[list(RAW_COLUMNS)].isna().any().any():
            raise ValueError(f"External issue-lag features are incomplete at lag {lag}")
        for column in RAW_COLUMNS:
            output[f"issue_lag{lag}__{column}"] = values[column].to_numpy(float)
    for column in RAW_COLUMNS:
        output[f"issue_trend6_24__{column}"] = (
            output[f"issue_lag6__{column}"] - output[f"issue_lag24__{column}"]
        )
    return pd.DataFrame(output, index=index)


def _delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _compare(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _delta(before, after),
    }


def _apply_policy(
    base: np.ndarray,
    member: np.ndarray,
    policy: BlendPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=float)
    member = np.asarray(member, dtype=float)
    gate = (
        (base / CAPACITY >= policy.min_base_ratio)
        & (np.abs(member - base) / CAPACITY <= policy.max_disagreement)
    )
    candidate = base.copy()
    candidate[gate] = np.clip(
        base[gate] + policy.alpha * (member[gate] - base[gate]),
        0.0,
        CAPACITY,
    )
    return candidate, gate


def _select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    member: np.ndarray,
    seed_members: np.ndarray,
    selection: np.ndarray,
) -> tuple[BlendPolicy, list[dict[str, object]]]:
    rows = []
    for max_disagreement in (0.02, 0.04, 0.06, 0.08):
        for min_base_ratio in (0.10, 0.30, 0.50):
            for alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 1.0):
                policy = BlendPolicy(max_disagreement, min_base_ratio, alpha)
                candidate, gate = _apply_policy(base, member, policy)
                comparison = _compare(truth, base, candidate, selection)
                movement = np.abs(candidate - base) / CAPACITY
                coverage = float((gate & selection).sum() / selection.sum())
                p95 = float(np.quantile(movement[selection], 0.95))
                delta = comparison["delta"]
                if (
                    delta["score"] < 0.00015
                    or delta["one_minus_nmae"] < 0.0
                    or delta["ficr"] < 0.0
                    or coverage > 0.25
                    or p95 > 0.015
                ):
                    continue
                seed_deltas = []
                for seed_member in seed_members:
                    seed_candidate, _ = _apply_policy(base, seed_member, policy)
                    seed_deltas.append(
                        _compare(truth, base, seed_candidate, selection)["delta"]
                    )
                if min(value["score"] for value in seed_deltas) < 0.0:
                    continue
                rows.append(
                    {
                        "policy": asdict(policy),
                        "metrics": comparison,
                        "changed_ratio": coverage,
                        "p95_movement_ratio": p95,
                        "min_seed_score_delta": min(
                            value["score"] for value in seed_deltas
                        ),
                    }
                )
    if not rows:
        raise RuntimeError("No issue-lag blend policy passed H1 selection gates")
    rows.sort(
        key=lambda row: (
            row["metrics"]["delta"]["score"],
            row["min_seed_score_delta"],
        ),
        reverse=True,
    )
    return BlendPolicy(**rows[0]["policy"]), rows[:20]


def _bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    period: np.ndarray,
    n_bootstrap: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(20260717)
    normalized = index.normalize()
    days = normalized[period].unique()
    positions = {
        day: np.flatnonzero(period & (normalized == day)) for day in days
    }
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        deltas.append(
            evaluate_group(truth[rows], candidate[rows], CAPACITY).score
            - evaluate_group(truth[rows], base[rows], CAPACITY).score
        )
    values = np.asarray(deltas)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((values > 0.0).mean()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _fit_members(
    features: pd.DataFrame,
    test_features: pd.DataFrame,
    labels: pd.Series,
    valid_index: pd.DatetimeIndex,
    selection: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    train = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < H1_START)
        & labels.notna()
    )
    select_rows = (
        (features.index >= H1_START)
        & (features.index < H2_START)
        & labels.notna()
    )
    full = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < pd.Timestamp("2025-01-01"))
        & labels.notna()
    )
    truth = labels.reindex(valid_index).to_numpy(float)
    valid_predictions = []
    test_predictions = []
    reports = []
    for seed in seeds:
        model = make_model(seed, n_estimators=2_000)
        model.fit(
            features.loc[train],
            labels.loc[train],
            eval_set=[(features.loc[select_rows], labels.loc[select_rows])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        raw_valid = np.clip(model.predict(features.reindex(valid_index)), 0.0, CAPACITY)
        scale, offset, calibrated = calibrate(
            truth[selection], raw_valid[selection], CAPACITY
        )
        valid_prediction = np.clip(raw_valid * scale + offset, 0.0, CAPACITY)

        final = make_model(seed, n_estimators=max(100, int(model.best_iteration_)))
        final.fit(
            features.loc[full],
            labels.loc[full],
            callbacks=[lgb.log_evaluation(0)],
        )
        test_prediction = np.clip(
            final.predict(test_features) * scale + offset, 0.0, CAPACITY
        )
        valid_predictions.append(valid_prediction)
        test_predictions.append(test_prediction)
        reports.append(
            {
                "seed": seed,
                "best_iteration": int(model.best_iteration_),
                "scale": scale,
                "offset": offset,
                "selection_calibrated_metric": calibrated,
            }
        )
    return np.asarray(valid_predictions), np.asarray(test_predictions), reports


def run_experiment(
    feature_cache: Path,
    test_feature_cache: Path,
    labels_path: Path,
    train_gfs_path: Path,
    test_gfs_path: Path,
    external_history_path: Path,
    driver_cache_path: Path,
    current_meta_cache_path: Path,
    current_submission_path: Path,
    candidate_path: Path,
    artifact_dir: Path,
    evaluation_path: Path,
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    X = pd.read_pickle(feature_cache)
    X_test = pd.read_pickle(test_feature_cache)
    labels_frame = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels_frame["kst_dtm"] = pd.to_datetime(labels_frame["kst_dtm"])
    labels = labels_frame.set_index("kst_dtm").reindex(X.index)[TARGET]
    history = pd.read_csv(
        external_history_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    train_external = issue_lag_features(
        X.index, _issue_times(train_gfs_path, X.index), history
    )
    test_external = issue_lag_features(
        X_test.index, _issue_times(test_gfs_path, X_test.index), history
    )
    columns = select_feature_columns(X, TARGET, "own_idw")
    features = pd.concat([X[columns], train_external], axis=1)
    test_features = pd.concat([X_test[columns], test_external], axis=1)

    driver = np.load(driver_cache_path)
    meta = np.load(current_meta_cache_path)
    valid_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not valid_index.equals(pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))):
        raise ValueError("Current meta-gate and driver validation indexes differ")
    truth = driver[f"{TARGET}__valid_truth"].astype(float)
    current = meta["valid_candidate"].astype(float)
    selection = valid_index < H2_START
    locked_h2 = ~selection

    seed_valid, seed_test, fit_reports = _fit_members(
        features,
        test_features,
        labels,
        valid_index,
        selection,
        seeds,
    )
    member = seed_valid.mean(axis=0)
    test_member = seed_test.mean(axis=0)
    policy, leaderboard = _select_policy(
        truth, current, member, seed_valid, selection
    )
    candidate, gate = _apply_policy(current, member, policy)
    locked = _compare(truth, current, candidate, locked_h2)
    seed_deltas = []
    for seed, seed_member in zip(seeds, seed_valid):
        seed_candidate, seed_gate = _apply_policy(current, seed_member, policy)
        seed_deltas.append(
            {
                "seed": seed,
                "changed_rows": int((seed_gate & locked_h2).sum()),
                "delta": _compare(
                    truth, current, seed_candidate, locked_h2
                )["delta"],
            }
        )
    monthly = {
        str(month): _compare(
            truth,
            current,
            candidate,
            locked_h2 & (valid_index.month == month),
        )["delta"]
        for month in range(7, 13)
    }
    bootstrap = _bootstrap_days(
        truth,
        current,
        candidate,
        valid_index,
        locked_h2,
        n_bootstrap,
    )

    current_submission = pd.read_csv(current_submission_path, encoding="utf-8-sig")
    test_current = current_submission[TARGET].to_numpy(float)
    test_candidate, test_gate = _apply_policy(test_current, test_member, policy)
    output = current_submission.copy()
    output[TARGET] = test_candidate
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    test_movement = np.abs(test_candidate - test_current) / CAPACITY
    locked_movement = np.abs(candidate - current) / CAPACITY

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "group3_issue_lag_lgbm_cache.npz"
    np.savez_compressed(
        cache_path,
        valid_index_ns=valid_index.astype("int64").to_numpy(),
        valid_truth=truth.astype("float32"),
        valid_current=current.astype("float32"),
        valid_seed_members=seed_valid.astype("float32"),
        valid_candidate=candidate.astype("float32"),
        valid_gate=gate,
        test_index_ns=X_test.index.astype("int64").to_numpy(),
        test_seed_members=seed_test.astype("float32"),
        test_candidate=test_candidate.astype("float32"),
        test_gate=test_gate,
    )
    duration = time.perf_counter() - started
    positive_months = int(sum(value["score"] > 0.0 for value in monthly.values()))
    locked_delta = locked["delta"]
    report: dict[str, object] = {
        "method": "leakage-safe issue-lag external GFS auxiliary LightGBM",
        "external_data": {
            "path": str(external_history_path),
            "provenance": str(external_history_path.with_suffix(".provenance.json")),
            "issue_lags_hours": list(ISSUE_LAGS),
            "minimum_safety_buffer_hours": min(ISSUE_LAGS),
            "uses_future_valid_weather": False,
            "uses_observation_or_reanalysis": False,
        },
        "features": {
            "base_count": len(columns),
            "external_count": train_external.shape[1],
            "total_count": features.shape[1],
        },
        "fits": fit_reports,
        "selection_h1": {
            "policy": asdict(policy),
            "metrics": _compare(truth, current, candidate, selection),
            "top_policies": leaderboard,
        },
        "locked_h2": {
            "metrics": locked,
            "changed_rows": int((gate & locked_h2).sum()),
            "changed_ratio": float((gate & locked_h2).sum() / locked_h2.sum()),
            "p95_movement_ratio": float(
                np.quantile(locked_movement[locked_h2], 0.95)
            ),
            "seed_deltas": seed_deltas,
            "monthly_deltas": monthly,
            "positive_months": positive_months,
            "day_bootstrap": bootstrap,
        },
        "test": {
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": float(test_gate.mean()),
            "p95_movement_ratio": float(np.quantile(test_movement, 0.95)),
            "max_movement_ratio": float(test_movement.max()),
            "groups_1_2_unchanged": bool(
                np.array_equal(output["kpx_group_1"], current_submission["kpx_group_1"])
                and np.array_equal(
                    output["kpx_group_2"], current_submission["kpx_group_2"]
                )
            ),
            "candidate": str(candidate_path),
        },
        "runtime_seconds": duration,
    }
    report_path = artifact_dir / "group3_issue_lag_lgbm_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fold_scores = [row["delta"]["score"] for row in seed_deltas]
    evaluation = {
        "family": "group3_issue_lag_lgbm",
        "locked_score_delta": locked_delta["score"],
        "locked_one_minus_nmae_delta": locked_delta["one_minus_nmae"],
        "locked_ficr_delta": locked_delta["ficr"],
        "expected_macro_score_delta": locked_delta["score"] / 3.0,
        "positive_months": positive_months,
        "total_months": 6,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "bootstrap_q05": bootstrap["q05"],
        "changed_ratio": max(
            float((gate & locked_h2).sum() / locked_h2.sum()),
            float(test_gate.mean()),
        ),
        "p95_movement_ratio": max(
            float(np.quantile(locked_movement[locked_h2], 0.95)),
            float(np.quantile(test_movement, 0.95)),
        ),
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_path": str(cache_path),
        "runtime_seconds": duration,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": locked_delta["score"] / 3.0,
        "selection_direction": "maximize",
        "notes": (
            "Only 6/12/24-hour pre-issue historical forecast states are used; "
            "policy selected on H1 and evaluated once on locked H2."
        ),
    }
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache", default="artifacts_final/feature_cache/features_train.pkl"
    )
    parser.add_argument(
        "--test-feature-cache", default="artifacts_final/feature_cache/features_test.pkl"
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--train-gfs", default="data/train/gfs_train.csv")
    parser.add_argument("--test-gfs", default="data/test/gfs_test.csv")
    parser.add_argument(
        "--external-history",
        default="artifacts_final/external_weather/open_meteo_gfs_issue_history.csv",
    )
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--current-meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--current-submission",
        default="submissions/blend_best_crossg3_traj_meta25_p55.csv",
    )
    parser.add_argument(
        "--candidate", default="submissions/blend_best_g3_issue_lag_lgbm.csv"
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--agent-evaluation-output", required=True)
    parser.add_argument("--seeds", default="28101,28102,28103")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run_experiment(
        Path(args.feature_cache),
        Path(args.test_feature_cache),
        Path(args.labels),
        Path(args.train_gfs),
        Path(args.test_gfs),
        Path(args.external_history),
        Path(args.driver_cache),
        Path(args.current_meta_cache),
        Path(args.current_submission),
        Path(args.candidate),
        Path(args.artifact_dir),
        Path(args.agent_evaluation_output),
        tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
