from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.group3_issue_lag_lgbm import (
    CAPACITY,
    H2_START,
    TARGET,
    _bootstrap_days,
    _compare,
    _fit_members,
    _issue_times,
    issue_lag_features,
)
from train import select_feature_columns


@dataclass(frozen=True)
class ConsensusPolicy:
    max_disagreement: float
    min_base_ratio: float
    max_seed_std_ratio: float
    coverage: float
    alpha: float


def apply_consensus(
    base: np.ndarray,
    seed_members: np.ndarray,
    period: np.ndarray,
    policy: ConsensusPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=float)
    seed_members = np.asarray(seed_members, dtype=float)
    mean_member = seed_members.mean(axis=0)
    std_ratio = seed_members.std(axis=0) / CAPACITY
    seed_delta = seed_members - base[None, :]
    unanimous = np.all(seed_delta > 0.0, axis=0) | np.all(seed_delta < 0.0, axis=0)
    action = (
        np.asarray(period, dtype=bool)
        & unanimous
        & (base / CAPACITY >= policy.min_base_ratio)
        & (np.abs(mean_member - base) / CAPACITY <= policy.max_disagreement)
        & (std_ratio <= policy.max_seed_std_ratio)
    )
    positions = np.flatnonzero(action)
    count = min(len(positions), int(np.floor(policy.coverage * np.sum(period))))
    gate = np.zeros(len(base), dtype=bool)
    if count > 0:
        confidence = (
            np.abs(mean_member[positions] - base[positions]) / CAPACITY
        ) / (std_ratio[positions] + 0.002)
        selected = positions[np.argpartition(confidence, -count)[-count:]]
        gate[selected] = True
    candidate = base.copy()
    candidate[gate] = np.clip(
        base[gate] + policy.alpha * (mean_member[gate] - base[gate]),
        0.0,
        CAPACITY,
    )
    return candidate, gate


def select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    seed_members: np.ndarray,
    selection: np.ndarray,
) -> tuple[ConsensusPolicy, list[dict[str, object]]]:
    rows = []
    for max_disagreement in (0.02, 0.04, 0.06):
        for min_base_ratio in (0.10, 0.30, 0.50):
            for max_seed_std_ratio in (0.0025, 0.005, 0.01, 0.02):
                for coverage in (0.01, 0.02, 0.03):
                    for alpha in (0.25, 0.50, 0.75, 1.0):
                        policy = ConsensusPolicy(
                            max_disagreement,
                            min_base_ratio,
                            max_seed_std_ratio,
                            coverage,
                            alpha,
                        )
                        candidate, gate = apply_consensus(
                            base, seed_members, selection, policy
                        )
                        comparison = _compare(truth, base, candidate, selection)
                        delta = comparison["delta"]
                        if (
                            delta["score"] < 0.00015
                            or delta["one_minus_nmae"] < 0.0
                            or delta["ficr"] < 0.0
                        ):
                            continue
                        seed_deltas = []
                        for seed_member in seed_members:
                            seed_candidate = base.copy()
                            seed_candidate[gate] = np.clip(
                                base[gate]
                                + policy.alpha * (seed_member[gate] - base[gate]),
                                0.0,
                                CAPACITY,
                            )
                            seed_deltas.append(
                                _compare(
                                    truth, base, seed_candidate, selection
                                )["delta"]
                            )
                        if min(value["score"] for value in seed_deltas) < 0.0:
                            continue
                        movement = np.abs(candidate - base) / CAPACITY
                        rows.append(
                            {
                                "policy": asdict(policy),
                                "metrics": comparison,
                                "changed_ratio": float(
                                    (gate & selection).sum() / selection.sum()
                                ),
                                "p95_movement_ratio": float(
                                    np.quantile(movement[selection], 0.95)
                                ),
                                "min_seed_score_delta": min(
                                    value["score"] for value in seed_deltas
                                ),
                            }
                        )
    if not rows:
        raise RuntimeError("No consensus policy passed H1 selection gates")
    rows.sort(
        key=lambda row: (
            row["metrics"]["delta"]["score"],
            row["min_seed_score_delta"],
        ),
        reverse=True,
    )
    return ConsensusPolicy(**rows[0]["policy"]), rows[:20]


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
    policy, leaderboard = select_policy(truth, current, seed_valid, selection)
    candidate, gate = apply_consensus(current, seed_valid, locked_h2, policy)
    locked = _compare(truth, current, candidate, locked_h2)
    seed_deltas = []
    for seed, seed_member in zip(seeds, seed_valid):
        seed_candidate = current.copy()
        seed_candidate[gate] = np.clip(
            current[gate] + policy.alpha * (seed_member[gate] - current[gate]),
            0.0,
            CAPACITY,
        )
        seed_deltas.append(
            {
                "seed": seed,
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
    test_period = np.ones(len(test_current), dtype=bool)
    test_candidate, test_gate = apply_consensus(
        test_current, seed_test, test_period, policy
    )
    output = current_submission.copy()
    output[TARGET] = test_candidate
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "group3_issue_lag_consensus_cache.npz"
    np.savez_compressed(
        cache_path,
        valid_index_ns=valid_index.astype("int64").to_numpy(),
        valid_seed_members=seed_valid.astype("float32"),
        valid_candidate=candidate.astype("float32"),
        valid_gate=gate,
        test_seed_members=seed_test.astype("float32"),
        test_candidate=test_candidate.astype("float32"),
        test_gate=test_gate,
    )
    locked_movement = np.abs(candidate - current) / CAPACITY
    test_movement = np.abs(test_candidate - test_current) / CAPACITY
    positive_months = int(sum(value["score"] > 0.0 for value in monthly.values()))
    duration = time.perf_counter() - started
    report: dict[str, object] = {
        "method": "three-seed consensus child of pre-issue weather-state LightGBM",
        "parent_run_id": 6,
        "parent_failure": (
            "Reduce the rejected 13.0% parent coverage by at least 75% and require "
            "unanimous seed direction before moving a row."
        ),
        "fits": fit_reports,
        "selection_h1": {
            "policy": asdict(policy),
            "top_policies": leaderboard,
        },
        "locked_h2": {
            "metrics": locked,
            "changed_rows": int(gate.sum()),
            "changed_ratio": float(gate.mean()),
            "p95_movement_ratio": float(np.quantile(locked_movement[locked_h2], 0.95)),
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
                and np.array_equal(output["kpx_group_2"], current_submission["kpx_group_2"])
            ),
            "candidate": str(candidate_path),
        },
        "runtime_seconds": duration,
    }
    (artifact_dir / "group3_issue_lag_consensus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    locked_delta = locked["delta"]
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
        "changed_ratio": max(float(gate.mean()), float(test_gate.mean())),
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
        "notes": "H1-selected unanimous seed-consensus child; locked/test coverage capped at 3%.",
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
        "--candidate", default="submissions/blend_best_g3_issue_lag_consensus.csv"
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
