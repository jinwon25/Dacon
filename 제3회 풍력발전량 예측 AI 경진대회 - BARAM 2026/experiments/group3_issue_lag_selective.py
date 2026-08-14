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
class SelectivePolicy:
    """A small, confidence-gated injection of the issue-lag member."""

    max_disagreement: float
    min_base_ratio: float
    max_seed_std_ratio: float
    coverage: float
    alpha: float
    direction: str
    ranker: str


def _confidence(
    base: np.ndarray,
    seed_members: np.ndarray,
    ranker: str,
) -> np.ndarray:
    mean_member = seed_members.mean(axis=0)
    std_ratio = seed_members.std(axis=0) / CAPACITY
    abs_delta_ratio = np.abs(mean_member - base) / CAPACITY
    if ranker == "signal_to_dispersion":
        return abs_delta_ratio / (std_ratio + 0.002)
    if ranker == "consensus_margin":
        minimum_seed_delta = np.min(np.abs(seed_members - base[None, :]), axis=0)
        return minimum_seed_delta / CAPACITY / (std_ratio + 0.002)
    if ranker == "absolute_delta":
        return abs_delta_ratio
    raise ValueError(f"Unknown confidence ranker: {ranker}")


def apply_selective(
    base: np.ndarray,
    seed_members: np.ndarray,
    period: np.ndarray,
    policy: SelectivePolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a policy without using labels to choose rows within ``period``."""

    base = np.asarray(base, dtype=float)
    seed_members = np.asarray(seed_members, dtype=float)
    period = np.asarray(period, dtype=bool)
    mean_member = seed_members.mean(axis=0)
    seed_delta = seed_members - base[None, :]
    unanimous = np.all(seed_delta > 0.0, axis=0) | np.all(seed_delta < 0.0, axis=0)
    eligible = (
        period
        & unanimous
        & (base / CAPACITY >= policy.min_base_ratio)
        & (np.abs(mean_member - base) / CAPACITY <= policy.max_disagreement)
        & (seed_members.std(axis=0) / CAPACITY <= policy.max_seed_std_ratio)
    )
    if policy.direction == "up":
        eligible &= mean_member > base
    elif policy.direction == "down":
        eligible &= mean_member < base
    elif policy.direction != "all":
        raise ValueError(f"Unknown direction: {policy.direction}")

    positions = np.flatnonzero(eligible)
    count = min(len(positions), int(np.floor(policy.coverage * period.sum())))
    gate = np.zeros(len(base), dtype=bool)
    if count:
        confidence = _confidence(base, seed_members, policy.ranker)
        selected = positions[np.argpartition(confidence[positions], -count)[-count:]]
        gate[selected] = True
    candidate = base.copy()
    candidate[gate] = np.clip(
        base[gate] + policy.alpha * (mean_member[gate] - base[gate]),
        0.0,
        CAPACITY,
    )
    return candidate, gate


def _seed_deltas(
    truth: np.ndarray,
    base: np.ndarray,
    seed_members: np.ndarray,
    gate: np.ndarray,
    period: np.ndarray,
    alpha: float,
) -> list[dict[str, float]]:
    rows = []
    for seed_member in seed_members:
        candidate = base.copy()
        candidate[gate] = np.clip(
            base[gate] + alpha * (seed_member[gate] - base[gate]),
            0.0,
            CAPACITY,
        )
        rows.append(_compare(truth, base, candidate, period)["delta"])
    return rows


def select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    seed_members: np.ndarray,
    index: pd.DatetimeIndex,
    selection: np.ndarray,
) -> tuple[SelectivePolicy | None, list[dict[str, object]], int]:
    """Select only on H1 OOF, including monthly and seed stability gates."""

    accepted: list[dict[str, object]] = []
    evaluated = 0
    for max_disagreement in (0.02, 0.04, 0.06):
        for min_base_ratio in (0.10, 0.30, 0.50):
            for max_seed_std_ratio in (0.0025, 0.005, 0.01, 0.02):
                for coverage in (0.02, 0.03, 0.04, 0.05):
                    for alpha in (0.02, 0.03, 0.04, 0.05):
                        for direction in ("all", "up", "down"):
                            for ranker in (
                                "signal_to_dispersion",
                                "consensus_margin",
                                "absolute_delta",
                            ):
                                evaluated += 1
                                policy = SelectivePolicy(
                                    max_disagreement=max_disagreement,
                                    min_base_ratio=min_base_ratio,
                                    max_seed_std_ratio=max_seed_std_ratio,
                                    coverage=coverage,
                                    alpha=alpha,
                                    direction=direction,
                                    ranker=ranker,
                                )
                                candidate, gate = apply_selective(
                                    base, seed_members, selection, policy
                                )
                                comparison = _compare(
                                    truth, base, candidate, selection
                                )
                                delta = comparison["delta"]
                                if (
                                    gate.sum() < 10
                                    or delta["score"] <= 0.0
                                    or delta["one_minus_nmae"] < 0.0
                                    or delta["ficr"] < 0.0
                                ):
                                    continue
                                seed_deltas = _seed_deltas(
                                    truth,
                                    base,
                                    seed_members,
                                    gate,
                                    selection,
                                    alpha,
                                )
                                min_seed_delta = min(
                                    row["score"] for row in seed_deltas
                                )
                                if min_seed_delta < 0.0:
                                    continue
                                monthly = [
                                    _compare(
                                        truth,
                                        base,
                                        candidate,
                                        selection & (index.month == month),
                                    )["delta"]
                                    for month in range(1, 7)
                                ]
                                negative_months = sum(
                                    row["score"] < -1e-12 for row in monthly
                                )
                                positive_months = sum(
                                    row["score"] > 1e-12 for row in monthly
                                )
                                if negative_months > 1 or positive_months < 2:
                                    continue
                                accepted.append(
                                    {
                                        "policy": asdict(policy),
                                        "metrics": comparison,
                                        "changed_rows": int(gate.sum()),
                                        "changed_ratio": float(
                                            gate.sum() / selection.sum()
                                        ),
                                        "min_seed_score_delta": min_seed_delta,
                                        "positive_months": positive_months,
                                        "negative_months": negative_months,
                                    }
                                )
    accepted.sort(
        key=lambda row: (
            row["negative_months"] == 0,
            row["metrics"]["delta"]["score"],
            row["min_seed_score_delta"],
        ),
        reverse=True,
    )
    if not accepted:
        return None, [], evaluated
    return SelectivePolicy(**accepted[0]["policy"]), accepted[:20], evaluated


def _load_or_fit_members(
    member_cache_path: Path,
    feature_cache: Path,
    test_feature_cache: Path,
    labels_path: Path,
    train_gfs_path: Path,
    test_gfs_path: Path,
    external_history_path: Path,
    driver_cache_path: Path,
    current_meta_cache_path: Path,
    seeds: tuple[int, ...],
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    driver = np.load(driver_cache_path)
    meta = np.load(current_meta_cache_path)
    valid_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not valid_index.equals(pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))):
        raise ValueError("Current meta-gate and driver validation indexes differ")
    truth = driver[f"{TARGET}__valid_truth"].astype(float)
    current = meta["valid_candidate"].astype(float)
    if member_cache_path.exists():
        cache = np.load(member_cache_path)
        cached_index = pd.DatetimeIndex(pd.to_datetime(cache["valid_index_ns"]))
        if not cached_index.equals(valid_index):
            raise ValueError("Issue-lag member cache validation index differs")
        return (
            valid_index,
            truth,
            current,
            cache["valid_seed_members"].astype(float),
            cache["test_seed_members"].astype(float),
            json.loads(str(cache["fit_reports_json"].item())),
        )

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
    selection = valid_index < H2_START
    seed_valid, seed_test, fit_reports = _fit_members(
        features,
        test_features,
        labels,
        valid_index,
        selection,
        seeds,
    )
    member_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        member_cache_path,
        valid_index_ns=valid_index.astype("int64").to_numpy(),
        valid_seed_members=seed_valid.astype("float32"),
        test_seed_members=seed_test.astype("float32"),
        fit_reports_json=json.dumps(fit_reports, ensure_ascii=False),
    )
    return valid_index, truth, current, seed_valid, seed_test, fit_reports


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
    member_cache_path: Path,
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    (
        valid_index,
        truth,
        current,
        seed_valid,
        seed_test,
        fit_reports,
    ) = _load_or_fit_members(
        member_cache_path,
        feature_cache,
        test_feature_cache,
        labels_path,
        train_gfs_path,
        test_gfs_path,
        external_history_path,
        driver_cache_path,
        current_meta_cache_path,
        seeds,
    )
    selection = valid_index < H2_START
    locked = ~selection
    policy, leaderboard, evaluated = select_policy(
        truth, current, seed_valid, valid_index, selection
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if policy is None:
        report: dict[str, object] = {
            "method": "2-5% selective issue-lag/Open-Meteo injection",
            "accepted": False,
            "reason": "No policy passed H1 OOF component, seed, and monthly gates.",
            "evaluated_policies": evaluated,
            "candidate_created": False,
            "runtime_seconds": time.perf_counter() - started,
        }
        (artifact_dir / "group3_issue_lag_selective_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    locked_candidate, locked_gate = apply_selective(
        current, seed_valid, locked, policy
    )
    locked_metrics = _compare(truth, current, locked_candidate, locked)
    locked_seed_deltas = _seed_deltas(
        truth,
        current,
        seed_valid,
        locked_gate,
        locked,
        policy.alpha,
    )
    monthly = {
        str(month): _compare(
            truth,
            current,
            locked_candidate,
            locked & (valid_index.month == month),
        )["delta"]
        for month in range(7, 13)
    }
    bootstrap = _bootstrap_days(
        truth,
        current,
        locked_candidate,
        valid_index,
        locked,
        n_bootstrap,
    )

    submission = pd.read_csv(current_submission_path, encoding="utf-8-sig")
    test_current = submission[TARGET].to_numpy(float)
    test_candidate, test_gate = apply_selective(
        test_current,
        seed_test,
        np.ones(len(test_current), dtype=bool),
        policy,
    )
    locked_delta = locked_metrics["delta"]
    positive_months = sum(row["score"] > 1e-12 for row in monthly.values())
    negative_months = sum(row["score"] < -1e-12 for row in monthly.values())
    min_seed_score_delta = min(
        row["score"] for row in locked_seed_deltas
    )
    locked_coverage = float(locked_gate.sum() / locked.sum())
    test_coverage = float(test_gate.mean())
    acceptance_checks = {
        "locked_score_positive": locked_delta["score"] > 0.0,
        "locked_nmae_nonnegative": locked_delta["one_minus_nmae"] >= 0.0,
        "locked_ficr_nonnegative": locked_delta["ficr"] >= 0.0,
        "all_seed_scores_nonnegative": min_seed_score_delta >= 0.0,
        "at_least_three_positive_months": positive_months >= 3,
        "at_most_one_negative_month": negative_months <= 1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.70,
        "bootstrap_q05_bounded": bootstrap["q05"] >= -0.0005,
        "coverage_within_five_percent": max(locked_coverage, test_coverage) <= 0.05,
        "coverage_shift_bounded": abs(test_coverage - locked_coverage) <= 0.02,
    }
    accepted = all(acceptance_checks.values())
    if accepted:
        output = submission.copy()
        output[TARGET] = test_candidate
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    cache_path = artifact_dir / "group3_issue_lag_selective_cache.npz"
    np.savez_compressed(
        cache_path,
        valid_index_ns=valid_index.astype("int64").to_numpy(),
        valid_truth=truth.astype("float32"),
        valid_current=current.astype("float32"),
        valid_candidate=locked_candidate.astype("float32"),
        valid_gate=locked_gate,
        test_candidate=test_candidate.astype("float32"),
        test_gate=test_gate,
    )
    report = {
        "method": "H1-selected, H2-locked 2-5% issue-lag/Open-Meteo injection",
        "accepted": accepted,
        "acceptance_checks": acceptance_checks,
        "candidate_created": accepted,
        "candidate": str(candidate_path) if accepted else None,
        "evaluated_policies": evaluated,
        "selected_policy": asdict(policy),
        "selection_h1": leaderboard[0],
        "selection_top_policies": leaderboard,
        "locked_h2": {
            "metrics": locked_metrics,
            "changed_rows": int(locked_gate.sum()),
            "changed_ratio": locked_coverage,
            "seed_deltas": locked_seed_deltas,
            "monthly_deltas": monthly,
            "positive_months": positive_months,
            "negative_months": negative_months,
            "day_bootstrap": bootstrap,
        },
        "test": {
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": test_coverage,
            "groups_1_2_unchanged": True,
        },
        "fits": fit_reports,
        "member_cache": str(member_cache_path),
        "oof_path": str(cache_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    (artifact_dir / "group3_issue_lag_selective_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
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
        "--candidate",
        default="submissions/blend_best_g3_issue_lag_selective_2to5.csv",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts_final/issue_lag_selective"
    )
    parser.add_argument(
        "--member-cache",
        default="artifacts_final/issue_lag_selective/member_cache.npz",
    )
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
        Path(args.member_cache),
        tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
