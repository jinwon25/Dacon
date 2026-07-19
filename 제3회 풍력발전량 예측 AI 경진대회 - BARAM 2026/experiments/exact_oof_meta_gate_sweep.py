from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from experiments.exact_oof_meta_gate import (
    CAPACITY,
    H2_START,
    META_SEEDS,
    Q2_START,
    TARGET,
    _bootstrap_days,
    _current_prediction,
    _evaluate_period,
    _fit_transfer_member,
    actionable_mask,
    apply_meta_gate,
    fit_probabilities,
    meta_features,
    settlement_benefit_labels,
)


REFERENCE_THRESHOLD = 0.55
REFERENCE_ALPHA = 0.25


@dataclass(frozen=True)
class Policy:
    threshold: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return {"threshold": self.threshold, "alpha": self.alpha}


def default_thresholds() -> tuple[float, ...]:
    return tuple(float(round(value, 3)) for value in np.arange(0.50, 0.651, 0.005))


def default_alphas() -> tuple[float, ...]:
    return tuple(float(round(value, 3)) for value in np.arange(0.10, 0.501, 0.025))


def evaluate_policy(
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    action: np.ndarray,
    probability: np.ndarray,
    seed_probabilities: list[np.ndarray],
    period: np.ndarray,
    policy: Policy,
    timestamps: pd.DatetimeIndex,
) -> dict[str, object]:
    candidate, gate = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=policy.threshold,
        extra_alpha=policy.alpha,
    )
    metrics = _evaluate_period(truth, current, candidate, period)
    seed_score_deltas: list[float] = []
    for seed_probability in seed_probabilities:
        seed_candidate, _ = apply_meta_gate(
            current,
            member,
            action,
            seed_probability,
            threshold=policy.threshold,
            extra_alpha=policy.alpha,
        )
        seed_score_deltas.append(
            float(_evaluate_period(truth, current, seed_candidate, period)["delta"]["score"])
        )
    monthly_score_deltas: dict[str, float] = {}
    for month in sorted(set(timestamps[period].month)):
        month_mask = period & (timestamps.month == month)
        # The lineage contains a boundary timestamp on each side of the split
        # (2024-07-01 00:00 and 2025-01-01 00:00). Do not count those single-row
        # slivers as additional validation months.
        if int(month_mask.sum()) < 24:
            continue
        monthly_score_deltas[str(month)] = float(
            _evaluate_period(truth, current, candidate, month_mask)["delta"]["score"]
        )
    movement = candidate - current
    return {
        "policy": policy.to_dict(),
        "changed_rows": int((gate & period).sum()),
        "mean_absolute_movement_kwh": float(np.abs(movement[period]).mean()),
        "metrics": metrics,
        "seed_score_deltas": seed_score_deltas,
        "min_seed_score_delta": float(min(seed_score_deltas)),
        "positive_seed_fraction": float(np.mean(np.asarray(seed_score_deltas) > 0.0)),
        "monthly_score_deltas": monthly_score_deltas,
        "months_improved": int(sum(value > 0.0 for value in monthly_score_deltas.values())),
    }


def select_policy(records: list[dict[str, object]]) -> dict[str, object]:
    """Select only on the development split, favoring seed-stable settlement gains."""
    eligible = []
    for record in records:
        delta = record["metrics"]["delta"]
        if (
            record["min_seed_score_delta"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and record["months_improved"] >= 2
        ):
            eligible.append(record)
    if not eligible:
        raise RuntimeError("No policy passed the development robustness contract")
    # The worst seed is the primary objective. Ensemble score breaks ties, followed by
    # the smaller intervention. This avoids choosing a brittle FICR boundary hit.
    return max(
        eligible,
        key=lambda record: (
            record["min_seed_score_delta"],
            record["metrics"]["delta"]["score"],
            -record["mean_absolute_movement_kwh"],
        ),
    )


def _prepare_validation(
    labels_path: Path, driver_cache_path: Path
) -> tuple[
    pd.DataFrame,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    cache = np.load(driver_cache_path, allow_pickle=True)
    index = pd.DatetimeIndex(pd.to_datetime(cache[f"{TARGET}__valid_index_ns"]))
    truth = labels.reindex(index)[TARGET].to_numpy(dtype=float)
    group_1 = cache["kpx_group_1__exact_base"].astype(float)
    group_2 = cache["kpx_group_2__exact_base"].astype(float)
    base = cache[f"{TARGET}__exact_base"].astype(float)
    member = _fit_transfer_member(labels, group_1, group_2, index, (2023,), 51_000)
    current = _current_prediction(group_1, group_2, base, member, index)
    features = meta_features(group_1, group_2, base, current, member, index)
    action = actionable_mask(group_1, group_2, base, member) & (truth >= 0.10 * CAPACITY)
    return labels, index, truth, group_1, group_2, base, member, current, features, action


def _make_submission(
    labels: pd.DataFrame,
    validation_features: np.ndarray,
    validation_action: np.ndarray,
    validation_truth: np.ndarray,
    validation_current: np.ndarray,
    validation_member: np.ndarray,
    validation_index: pd.DatetimeIndex,
    pre_cross_test_path: Path,
    current_test_path: Path,
    output_path: Path,
    policy: Policy,
) -> dict[str, object]:
    pre_cross_test = pd.read_csv(pre_cross_test_path, encoding="utf-8-sig")
    current_test = pd.read_csv(current_test_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(pd.to_datetime(current_test["forecast_kst_dtm"]))
    test_group_1 = pre_cross_test["kpx_group_1"].to_numpy(dtype=float)
    test_group_2 = pre_cross_test["kpx_group_2"].to_numpy(dtype=float)
    test_base = pre_cross_test[TARGET].to_numpy(dtype=float)
    test_current = current_test[TARGET].to_numpy(dtype=float)
    test_member = _fit_transfer_member(
        labels, test_group_1, test_group_2, test_index, (2023, 2024), 52_000
    )
    test_features = meta_features(
        test_group_1, test_group_2, test_base, test_current, test_member, test_index
    )
    test_action = actionable_mask(test_group_1, test_group_2, test_base, test_member)
    full_train = validation_index < test_index.min()
    benefit = settlement_benefit_labels(
        validation_truth, validation_current, validation_member, full_train
    )
    test_probabilities = []
    for seed in META_SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=50,
            max_features=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(validation_features[full_train & validation_action], benefit[full_train & validation_action])
        test_probabilities.append(model.predict_proba(test_features)[:, 1])
    test_probability = np.mean(test_probabilities, axis=0)
    test_candidate, test_gate = apply_meta_gate(
        test_current,
        test_member,
        test_action,
        test_probability,
        threshold=policy.threshold,
        extra_alpha=policy.alpha,
    )
    output = current_test.copy()
    output[TARGET] = test_candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    output_bytes = output_path.read_bytes()
    movement = test_candidate - test_current
    return {
        "output": str(output_path),
        "file_bytes": len(output_bytes),
        "file_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "changed_rows": int(test_gate.sum()),
        "changed_ratio": float(test_gate.mean()),
        "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
        "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
        "max_absolute_movement_kwh": float(np.abs(movement).max()),
        "groups_1_2_unchanged": bool(
            np.array_equal(output["kpx_group_1"], current_test["kpx_group_1"])
            and np.array_equal(output["kpx_group_2"], current_test["kpx_group_2"])
        ),
    }


def run_sweep(
    labels_path: Path,
    driver_cache_path: Path,
    pre_cross_test_path: Path,
    current_test_path: Path,
    output_path: Path,
    artifact_dir: Path,
    n_bootstrap: int,
    write_submission_if_qualified: bool,
) -> dict[str, object]:
    (
        labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        features,
        action,
    ) = _prepare_validation(labels_path, driver_cache_path)
    q1 = index < Q2_START
    q2 = (index >= Q2_START) & (index < H2_START)
    h1 = index < H2_START
    h2 = index >= H2_START

    q1_benefit = settlement_benefit_labels(truth, current, member, q1)
    q2_probability, q2_seed_probabilities = fit_probabilities(
        features, q1_benefit, q1 & action
    )
    policies = [
        Policy(threshold, alpha)
        for threshold in default_thresholds()
        for alpha in default_alphas()
    ]
    development_records = [
        evaluate_policy(
            truth,
            current,
            member,
            action,
            q2_probability,
            q2_seed_probabilities,
            q2,
            policy,
            index,
        )
        for policy in policies
    ]
    selected_development = select_policy(development_records)
    selected_policy = Policy(**selected_development["policy"])

    h1_benefit = settlement_benefit_labels(truth, current, member, h1)
    h2_probability, h2_seed_probabilities = fit_probabilities(
        features, h1_benefit, h1 & action
    )
    reference_policy = Policy(REFERENCE_THRESHOLD, REFERENCE_ALPHA)
    reference_locked = evaluate_policy(
        truth,
        current,
        member,
        action,
        h2_probability,
        h2_seed_probabilities,
        h2,
        reference_policy,
        index,
    )
    selected_locked = evaluate_policy(
        truth,
        current,
        member,
        action,
        h2_probability,
        h2_seed_probabilities,
        h2,
        selected_policy,
        index,
    )
    selected_candidate, _ = apply_meta_gate(
        current,
        member,
        action,
        h2_probability,
        threshold=selected_policy.threshold,
        extra_alpha=selected_policy.alpha,
    )
    reference_candidate, _ = apply_meta_gate(
        current,
        member,
        action,
        h2_probability,
        threshold=reference_policy.threshold,
        extra_alpha=reference_policy.alpha,
    )
    selected_bootstrap = _bootstrap_days(
        truth, current, selected_candidate, index, h2, n_bootstrap, seed=20260718
    )
    reference_bootstrap = _bootstrap_days(
        truth, current, reference_candidate, index, h2, n_bootstrap, seed=20260718
    )
    incremental_metrics = _evaluate_period(
        truth, reference_candidate, selected_candidate, h2
    )
    incremental_monthly: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        month_mask = h2 & (index.month == month)
        incremental_monthly[str(month)] = _evaluate_period(
            truth, reference_candidate, selected_candidate, month_mask
        )["delta"]
    incremental_bootstrap = _bootstrap_days(
        truth,
        reference_candidate,
        selected_candidate,
        index,
        h2,
        n_bootstrap,
        seed=20260718,
    )

    selected_delta = selected_locked["metrics"]["delta"]
    reference_delta = reference_locked["metrics"]["delta"]
    qualified = bool(
        selected_delta["score"] >= reference_delta["score"] + 0.00005
        and selected_delta["one_minus_nmae"] > 0.0
        and selected_delta["ficr"] > 0.0
        and selected_locked["positive_seed_fraction"] == 1.0
        and selected_locked["months_improved"] >= 4
        and selected_bootstrap["positive_fraction"] >= reference_bootstrap["positive_fraction"]
        and selected_bootstrap["q05"] >= reference_bootstrap["q05"]
    )
    submission = None
    if qualified and write_submission_if_qualified:
        submission = _make_submission(
            labels,
            features,
            action,
            truth,
            current,
            member,
            index,
            pre_cross_test_path,
            current_test_path,
            output_path,
            selected_policy,
        )

    top_development = sorted(
        development_records,
        key=lambda record: (
            record["min_seed_score_delta"], record["metrics"]["delta"]["score"]
        ),
        reverse=True,
    )[:20]
    report: dict[str, object] = {
        "method": "leakage-safe fine sweep of exact OOF settlement meta-gate",
        "selection_contract": {
            "development": "Q1 train -> Q2 select",
            "locked": "H1 train -> H2 evaluate selected policy once",
            "threshold_grid": list(default_thresholds()),
            "alpha_grid": list(default_alphas()),
            "policy_count": len(policies),
            "primary_objective": "maximum worst-seed Q2 score delta",
            "constraints": [
                "Q2 ensemble 1-NMAE and FICR deltas both positive",
                "all Q2 seed score deltas positive",
                "at least two of three Q2 months improve",
            ],
        },
        "selected_development": selected_development,
        "top_development": top_development,
        "locked_reference": reference_locked,
        "locked_selected": selected_locked,
        "locked_reference_bootstrap": reference_bootstrap,
        "locked_selected_bootstrap": selected_bootstrap,
        "locked_incremental_over_reference": {
            "reference_policy": reference_policy.to_dict(),
            "selected_policy": selected_policy.to_dict(),
            "metrics": incremental_metrics,
            "monthly_deltas": incremental_monthly,
            "positive_months": int(
                sum(row["score"] > 0.0 for row in incremental_monthly.values())
            ),
            "day_bootstrap": incremental_bootstrap,
        },
        "qualification": {
            "qualified": qualified,
            "required_score_margin_over_reference": 0.00005,
            "observed_score_margin_over_reference": float(
                selected_delta["score"] - reference_delta["score"]
            ),
            "submission_requested": write_submission_if_qualified,
            "submission_created": submission is not None,
        },
        "submission": submission,
        "decision": (
            "Create a submission candidate: the Q2-selected fine policy cleared the locked "
            "score, component, seed, month, and bootstrap gates."
            if submission is not None
            else "Do not create a submission candidate: the fine policy did not clear every locked robustness gate."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "meta_gate_policy_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--pre-cross-test", default="artifacts_final/lineage_inputs/base_pre_cross.csv")
    parser.add_argument(
        "--current-test",
        default="submissions/archive/blend_best_crossg3_traj5_consensus.csv",
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_crossg3_traj_meta_finesweep.csv"
    )
    parser.add_argument("--artifact-dir", default="artifacts_final/meta_gate_sweep")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()
    report = run_sweep(
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.pre_cross_test),
        Path(args.current_test),
        Path(args.output),
        Path(args.artifact_dir),
        args.n_bootstrap,
        args.write_submission_if_qualified,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
