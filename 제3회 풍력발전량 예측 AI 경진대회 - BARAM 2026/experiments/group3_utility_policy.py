from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from experiments.exact_oof_meta_gate import (
    H2_START,
    META_SEEDS,
    Q2_START,
    _bootstrap_days,
    _evaluate_period,
    _fit_transfer_member,
    meta_features,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from experiments.group3_physical_residual import build_physical_features
from experiments.spatiotemporal_consensus_promotion import _rolling_finesweep_base
from src.metrics import CAPACITY_KWH


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
ACTION_RATIOS = np.asarray((-0.010, -0.005, -0.0025, 0.0025, 0.005, 0.010))


@dataclass(frozen=True)
class Policy:
    min_samples_leaf: int
    minimum_advantage: float


def policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(leaf, advantage)
        for leaf in (40, 80, 160)
        for advantage in (0.0, 0.00025, 0.00050, 0.00100)
    )


def utility_delta_targets(
    truth: np.ndarray,
    current: np.ndarray,
    period: np.ndarray,
) -> np.ndarray:
    """Fully observed official-score contribution for each additive action."""
    truth = np.asarray(truth, dtype=float)
    current = np.asarray(current, dtype=float)
    period = np.asarray(period, dtype=bool)
    eligible = period & np.isfinite(truth) & (truth >= 0.10 * CAPACITY)
    if not eligible.any():
        raise ValueError("Utility target period has no eligible observations")
    count = int(eligible.sum())
    generation_sum = float(truth[eligible].sum())

    def contribution(prediction: np.ndarray) -> np.ndarray:
        error = np.abs(truth - prediction) / CAPACITY
        units = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
        return -0.5 * error + 0.5 * count * truth * units / (4.0 * generation_sum)

    baseline = contribution(current)
    targets = []
    for ratio in ACTION_RATIOS:
        prediction = np.clip(current + ratio * CAPACITY, 0.0, CAPACITY)
        targets.append(contribution(prediction) - baseline)
    return np.column_stack(targets)


def fit_action_models(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    min_samples_leaf: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    predictions = []
    for seed in META_SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed + 31_000,
        )
        model.fit(features[train_mask], targets[train_mask])
        predictions.append(model.predict(features))
    return np.mean(predictions, axis=0), predictions


def apply_policy(
    current: np.ndarray,
    predicted_advantages: np.ndarray,
    minimum_advantage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    predicted_advantages = np.asarray(predicted_advantages, dtype=float)
    if predicted_advantages.shape != (len(current), len(ACTION_RATIOS)):
        raise ValueError("Unexpected action-advantage matrix shape")
    best_action = np.argmax(predicted_advantages, axis=1)
    best_advantage = predicted_advantages[np.arange(len(current)), best_action]
    gate = (current >= 0.10 * CAPACITY) & (best_advantage >= minimum_advantage)
    action_ratio = np.zeros(len(current), dtype=float)
    action_ratio[gate] = ACTION_RATIOS[best_action[gate]]
    candidate = np.clip(current + action_ratio * CAPACITY, 0.0, CAPACITY)
    return candidate, gate, action_ratio


def evaluate_policy(
    truth: np.ndarray,
    current: np.ndarray,
    ensemble_advantages: np.ndarray,
    seed_advantages: list[np.ndarray],
    period: np.ndarray,
    index: pd.DatetimeIndex,
    policy: Policy,
) -> dict[str, object]:
    candidate, gate, action = apply_policy(
        current, ensemble_advantages, policy.minimum_advantage
    )
    comparison = _evaluate_period(truth, current, candidate, period)
    seed_deltas = []
    for seed_prediction in seed_advantages:
        seed_candidate, _, _ = apply_policy(
            current, seed_prediction, policy.minimum_advantage
        )
        seed_deltas.append(
            _evaluate_period(truth, current, seed_candidate, period)["delta"]
        )
    monthly = {}
    for month in sorted(set(index[period].month)):
        month_mask = period & (index.month == month)
        if int(month_mask.sum()) >= 24:
            monthly[str(month)] = _evaluate_period(
                truth, current, candidate, month_mask
            )["delta"]
    return {
        "policy": asdict(policy),
        "comparison": comparison,
        "changed_rows": int((gate & period).sum()),
        "changed_ratio": float((gate & period).sum() / max(int(period.sum()), 1)),
        "mean_absolute_movement_kwh": float(
            np.abs(candidate[period] - current[period]).mean()
        ),
        "action_counts": {
            str(ratio): int(((action == ratio) & period).sum())
            for ratio in ACTION_RATIOS
        },
        "seed_deltas": seed_deltas,
        "min_seed_score_delta": float(
            min(delta["score"] for delta in seed_deltas)
        ),
        "all_seed_score_positive": bool(
            all(delta["score"] > 0.0 for delta in seed_deltas)
        ),
        "monthly_deltas": monthly,
        "positive_months": int(
            sum(delta["score"] > 0.0 for delta in monthly.values())
        ),
    }


def select_development(records: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = []
    for record in records:
        delta = record["comparison"]["delta"]
        if (
            delta["score"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and record["all_seed_score_positive"]
            and record["positive_months"] >= 2
        ):
            eligible.append(record)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            record["min_seed_score_delta"],
            record["comparison"]["delta"]["score"],
            -record["mean_absolute_movement_kwh"],
        ),
    )


def _feature_matrix(
    raw_features: pd.DataFrame,
    index: pd.DatetimeIndex,
    meta: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    physical, _families = build_physical_features(raw_features)
    physical = physical.reindex(index)
    if physical.isna().any().any():
        raise ValueError("Physical policy features contain missing values")
    names = [f"meta_{i}" for i in range(meta.shape[1])] + list(physical.columns)
    matrix = np.column_stack([meta, physical.to_numpy(dtype=float)])
    if not np.isfinite(matrix).all():
        raise ValueError("Utility-policy feature matrix contains non-finite values")
    return matrix, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--train-features", default="artifacts_final/feature_cache/features_train.pkl"
    )
    parser.add_argument(
        "--test-features", default="artifacts_final/feature_cache/features_test.pkl"
    )
    parser.add_argument(
        "--pre-cross-test", default="artifacts_final/lineage_inputs/base_pre_cross.csv"
    )
    parser.add_argument(
        "--base-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_g3_utility_policy.csv"
    )
    parser.add_argument(
        "--report", default="artifacts_final/utility_policy/group3_utility_policy_report.json"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    labels, index, truth, rolling = _rolling_finesweep_base(
        Path(args.labels), Path(args.driver)
    )
    (
        _labels,
        prepared_index,
        _truth,
        group_1,
        group_2,
        base,
        member,
        _current,
        _meta,
        _action,
    ) = _prepare_validation(Path(args.labels), Path(args.driver))
    if not index.equals(prepared_index):
        raise ValueError("Rolling and prepared OOF indexes differ")
    meta = meta_features(group_1, group_2, base, rolling, member, index)
    raw_train = pd.read_pickle(args.train_features)
    features, feature_names = _feature_matrix(raw_train, index, meta)

    q1 = np.asarray(index < Q2_START)
    q2 = np.asarray((index >= Q2_START) & (index < H2_START))
    h1 = np.asarray(index < H2_START)
    h2 = np.asarray(index >= H2_START)
    q1_train = q1 & (truth >= 0.10 * CAPACITY) & (rolling >= 0.10 * CAPACITY)
    h1_train = h1 & (truth >= 0.10 * CAPACITY) & (rolling >= 0.10 * CAPACITY)

    q1_targets = utility_delta_targets(truth, rolling, q1)
    development = []
    q2_predictions: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}
    for leaf in sorted({policy.min_samples_leaf for policy in policies()}):
        q2_predictions[leaf] = fit_action_models(
            features, q1_targets, q1_train, leaf
        )
    for policy in policies():
        ensemble, seeds = q2_predictions[policy.min_samples_leaf]
        development.append(
            evaluate_policy(truth, rolling, ensemble, seeds, q2, index, policy)
        )
    selected = select_development(development)

    locked = None
    locked_bootstrap = None
    locked_candidate = None
    qualified = False
    if selected is not None:
        selected_policy = Policy(**selected["policy"])
        h1_targets = utility_delta_targets(truth, rolling, h1)
        h2_ensemble, h2_seeds = fit_action_models(
            features,
            h1_targets,
            h1_train,
            selected_policy.min_samples_leaf,
        )
        locked = evaluate_policy(
            truth,
            rolling,
            h2_ensemble,
            h2_seeds,
            h2,
            index,
            selected_policy,
        )
        locked_candidate, _, _ = apply_policy(
            rolling, h2_ensemble, selected_policy.minimum_advantage
        )
        locked_bootstrap = _bootstrap_days(
            truth,
            rolling,
            locked_candidate,
            index,
            h2,
            args.bootstrap,
            seed=20260718,
        )
        delta = locked["comparison"]["delta"]
        qualified = bool(
            delta["score"] >= 0.00015
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and locked["all_seed_score_positive"]
            and locked["positive_months"] >= 4
            and locked_bootstrap["positive_fraction"] >= 0.90
            and locked_bootstrap["q05"] >= -0.00025
        )

    submission = None
    if qualified and selected is not None:
        selected_policy = Policy(**selected["policy"])
        source = pd.read_csv(args.base_submission, encoding="utf-8-sig")
        test_index = pd.DatetimeIndex(pd.to_datetime(source["forecast_kst_dtm"]))
        pre_cross = pd.read_csv(args.pre_cross_test, encoding="utf-8-sig")
        if not test_index.equals(
            pd.DatetimeIndex(pd.to_datetime(pre_cross["forecast_kst_dtm"]))
        ):
            raise ValueError("Pre-cross and public-best test indexes differ")
        test_group_1 = source["kpx_group_1"].to_numpy(dtype=float)
        test_group_2 = source["kpx_group_2"].to_numpy(dtype=float)
        test_base = pre_cross[TARGET].to_numpy(dtype=float)
        test_current = source[TARGET].to_numpy(dtype=float)
        test_member = _fit_transfer_member(
            labels,
            test_group_1,
            test_group_2,
            test_index,
            (2023, 2024),
            52_000,
        )
        test_meta = meta_features(
            test_group_1,
            test_group_2,
            test_base,
            test_current,
            test_member,
            test_index,
        )
        raw_test = pd.read_pickle(args.test_features)
        test_features, test_feature_names = _feature_matrix(
            raw_test, test_index, test_meta
        )
        if feature_names != test_feature_names:
            raise ValueError("Train/test utility feature columns differ")
        full = np.asarray((truth >= 0.10 * CAPACITY) & (rolling >= 0.10 * CAPACITY))
        full_targets = utility_delta_targets(
            truth, rolling, np.ones(len(index), dtype=bool)
        )
        test_seed_predictions = []
        for seed in META_SEEDS:
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=selected_policy.min_samples_leaf,
                max_features=0.8,
                n_jobs=-1,
                random_state=seed + 31_000,
            )
            model.fit(features[full], full_targets[full])
            test_seed_predictions.append(model.predict(test_features))
        test_advantages = np.mean(test_seed_predictions, axis=0)
        test_candidate, test_gate, test_action = apply_policy(
            test_current, test_advantages, selected_policy.minimum_advantage
        )
        output = source.copy()
        output[TARGET] = test_candidate
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        payload = output_path.read_bytes()
        submission = {
            "output": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": int(len(output)),
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": float(test_gate.mean()),
            "mean_absolute_movement_kwh": float(
                np.abs(test_candidate - test_current).mean()
            ),
            "action_counts": {
                str(ratio): int((test_action == ratio).sum())
                for ratio in ACTION_RATIOS
            },
            "groups_1_2_unchanged": bool(
                np.array_equal(output["kpx_group_1"], source["kpx_group_1"])
                and np.array_equal(output["kpx_group_2"], source["kpx_group_2"])
            ),
        }

    report = {
        "method": "direct official-utility action policy on public-best group 3",
        "hypothesis": (
            "Predicting the official score contribution of bounded additive actions "
            "handles the 6%/8% FICR discontinuities better than residual-mean regression."
        ),
        "validation_contract": {
            "development": "Q1 train -> Q2 select",
            "locked": "H1 train -> H2 open once",
            "baseline": "rolling exact OOF equivalent of public finesweep",
            "actions_capacity_ratio": ACTION_RATIOS.tolist(),
            "policy_count": len(policies()),
            "feature_count": len(feature_names),
        },
        "development_selected": selected,
        "development_top": sorted(
            development,
            key=lambda item: item["comparison"]["delta"]["score"],
            reverse=True,
        )[:5],
        "locked_h2": locked,
        "locked_h2_bootstrap": locked_bootstrap,
        "qualified": qualified,
        "submission": submission,
        "decision": (
            "create direct-utility submission candidate"
            if qualified
            else "reject; no submission candidate"
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
