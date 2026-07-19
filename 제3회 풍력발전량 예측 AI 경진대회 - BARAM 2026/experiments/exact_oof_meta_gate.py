from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from experiments.cross_group_trajectory_smoothing import (
    _current_cross_group_prediction,
    smooth_group_3,
)
from experiments.cross_group_transfer import fit_predict_models, transfer_features
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
META_SEEDS = (1_300, 1_301, 1_302, 1_303, 1_304)


def meta_features(
    group_1: np.ndarray,
    group_2: np.ndarray,
    base: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    group_1_ratio = np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"]
    base_ratio = np.asarray(base, dtype=float) / CAPACITY
    current_ratio = np.asarray(current, dtype=float) / CAPACITY
    member_ratio = np.asarray(member, dtype=float) / CAPACITY
    delta = member_ratio - current_ratio
    hour = timestamps.hour.to_numpy()
    return np.column_stack(
        [
            group_1_ratio,
            group_2_ratio,
            (group_1_ratio + group_2_ratio) / 2.0,
            group_2_ratio - group_1_ratio,
            base_ratio,
            current_ratio,
            member_ratio,
            delta,
            np.abs(delta),
            np.abs(group_1_ratio - group_2_ratio),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ]
    )


def actionable_mask(
    group_1: np.ndarray,
    group_2: np.ndarray,
    base: np.ndarray,
    member: np.ndarray,
) -> np.ndarray:
    group_1_ratio = np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"]
    base = np.asarray(base, dtype=float)
    member = np.asarray(member, dtype=float)
    return (
        (np.abs(group_1_ratio - group_2_ratio) <= 0.08)
        & (np.abs(member - base) / CAPACITY <= 0.06)
        & (base >= 0.10 * CAPACITY)
    )


def settlement_benefit_labels(
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    period_mask: np.ndarray,
    step: float = 0.10,
) -> np.ndarray:
    """Label rows whose small move toward the member raises official utility."""
    truth = np.asarray(truth, dtype=float)
    current = np.asarray(current, dtype=float)
    member = np.asarray(member, dtype=float)
    period_mask = np.asarray(period_mask, dtype=bool)
    eligible = period_mask & np.isfinite(truth) & (truth >= 0.10 * CAPACITY)
    if not eligible.any():
        raise ValueError("The label period has no eligible observations")
    candidate = np.clip(current + step * (member - current), 0.0, CAPACITY)
    base_error = np.abs(truth - current) / CAPACITY
    candidate_error = np.abs(truth - candidate) / CAPACITY
    base_units = np.where(base_error <= 0.06, 4.0, np.where(base_error <= 0.08, 3.0, 0.0))
    candidate_units = np.where(
        candidate_error <= 0.06, 4.0, np.where(candidate_error <= 0.08, 3.0, 0.0)
    )
    # Multiplying the official score delta by the eligible sample count
    # preserves its sign and puts NMAE/FICR row contributions on one scale.
    contribution = -0.5 * (candidate_error - base_error)
    contribution += (
        0.5
        * int(eligible.sum())
        * truth
        * (candidate_units - base_units)
        / (4.0 * float(truth[eligible].sum()))
    )
    return contribution > 0.0


def fit_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    seeds: tuple[int, ...] = META_SEEDS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    probabilities = []
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=50,
            max_features=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(features[train_mask], labels[train_mask])
        probabilities.append(model.predict_proba(features)[:, 1])
    return np.mean(probabilities, axis=0), probabilities


def apply_meta_gate(
    current: np.ndarray,
    member: np.ndarray,
    action_mask: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.55,
    extra_alpha: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    member = np.asarray(member, dtype=float)
    gate = np.asarray(action_mask, dtype=bool) & (np.asarray(probability) >= threshold)
    output = current.copy()
    output[gate] = np.clip(
        current[gate] + extra_alpha * (member[gate] - current[gate]), 0.0, CAPACITY
    )
    return output, gate


def _delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _evaluate_period(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {"base": before.to_dict(), "candidate": after.to_dict(), "delta": _delta(before, after)}


def _fit_transfer_member(
    labels: pd.DataFrame,
    group_1: np.ndarray,
    group_2: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_years: tuple[int, ...],
    seed: int,
) -> np.ndarray:
    train = labels[labels.index.year.isin(train_years)].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    return np.clip(
        fit_predict_models(
            transfer_features(
                train["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"],
                train["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"],
                train.index,
            ),
            train[TARGET].to_numpy(dtype=float),
            transfer_features(
                np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"],
                np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"],
                timestamps,
            ),
            seed=seed,
            mode="base",
        ),
        0.0,
        CAPACITY,
    )


def _current_prediction(
    group_1: np.ndarray,
    group_2: np.ndarray,
    base: np.ndarray,
    member: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    cross = _current_cross_group_prediction(
        base,
        member,
        np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"],
        np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"],
    )
    smoothed, _ = smooth_group_3(
        group_1, group_2, cross, timestamps, alpha=0.05, max_delta_ratio=0.02
    )
    return smoothed


def _bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    timestamps: pd.DatetimeIndex,
    period: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    days = timestamps[period].normalize().unique()
    positions = {
        day: np.flatnonzero(period & (timestamps.normalize() == day)) for day in days
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
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def run_experiment(
    labels_path: Path,
    driver_cache_path: Path,
    pre_cross_test_path: Path,
    current_test_path: Path,
    output_path: Path,
    artifact_dir: Path,
    n_bootstrap: int,
) -> dict[str, object]:
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

    periods = {
        "q1": index < Q2_START,
        "q2": (index >= Q2_START) & (index < H2_START),
        "h1": index < H2_START,
        "h2": index >= H2_START,
    }
    split_reports: dict[str, object] = {}
    retained_probability = None
    retained_candidate = None
    retained_gate = None
    for name, train_name, evaluate_name in (
        ("inner_q1_to_q2", "q1", "q2"),
        ("locked_h1_to_h2", "h1", "h2"),
    ):
        train_period = periods[train_name]
        evaluate_period = periods[evaluate_name]
        benefit = settlement_benefit_labels(truth, current, member, train_period)
        probability, seed_probabilities = fit_probabilities(
            features, benefit, train_period & action
        )
        candidate, gate = apply_meta_gate(current, member, action, probability)
        seed_deltas = []
        for seed, seed_probability in zip(META_SEEDS, seed_probabilities):
            seed_candidate, seed_gate = apply_meta_gate(
                current, member, action, seed_probability
            )
            seed_metric = _evaluate_period(
                truth, current, seed_candidate, evaluate_period
            )
            seed_deltas.append(
                {
                    "seed": seed,
                    "changed_rows": int((seed_gate & evaluate_period).sum()),
                    "delta": seed_metric["delta"],
                }
            )
        split_reports[name] = {
            "train_rows": int((train_period & action).sum()),
            "positive_label_fraction": float(benefit[train_period & action].mean()),
            "changed_rows": int((gate & evaluate_period).sum()),
            "metrics": _evaluate_period(truth, current, candidate, evaluate_period),
            "seed_deltas": seed_deltas,
        }
        if name == "locked_h1_to_h2":
            retained_probability = probability
            retained_candidate = candidate
            retained_gate = gate

    if retained_candidate is None or retained_gate is None or retained_probability is None:
        raise RuntimeError("Locked validation split was not evaluated")
    monthly = {}
    for month in range(7, 13):
        mask = periods["h2"] & (index.month == month)
        monthly[str(month)] = _evaluate_period(truth, current, retained_candidate, mask)["delta"]
    bootstrap = _bootstrap_days(
        truth,
        current,
        retained_candidate,
        index,
        periods["h2"],
        n_bootstrap,
        seed=20260717,
    )

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
    full_train = index < test_index.min()
    full_benefit = settlement_benefit_labels(truth, current, member, full_train)
    all_features = np.vstack([features, test_features])
    full_probabilities = []
    for seed in META_SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=50,
            max_features=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(features[full_train & action], full_benefit[full_train & action])
        full_probabilities.append(model.predict_proba(all_features)[:, 1][len(features) :])
    test_probability = np.mean(full_probabilities, axis=0)
    test_candidate, test_gate = apply_meta_gate(
        test_current, test_member, test_action, test_probability
    )
    output = current_test.copy()
    output[TARGET] = test_candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    movement = test_candidate - test_current
    report: dict[str, object] = {
        "method": "exact OOF settlement-aware cross-group meta-gate",
        "policy": {
            "classifier": "ExtraTreesClassifier",
            "n_estimators_per_seed": 600,
            "min_samples_leaf": 50,
            "seeds": list(META_SEEDS),
            "label_step": 0.10,
            "probability_threshold": 0.55,
            "extra_alpha": 0.25,
        },
        "validation": split_reports,
        "locked_h2_monthly_deltas": monthly,
        "locked_h2_day_bootstrap": bootstrap,
        "final": {
            "train_rows": int((full_train & action).sum()),
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": float(test_gate.mean()),
            "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
            "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "max_absolute_movement_kwh": float(np.abs(movement).max()),
            "groups_1_2_unchanged": bool(
                np.array_equal(output["kpx_group_1"], current_test["kpx_group_1"])
                and np.array_equal(output["kpx_group_2"], current_test["kpx_group_2"])
            ),
            "output": str(output_path),
        },
        "decision": (
            "Controlled public probe only: exact OOF Q2/H2 and all seed deltas are positive, "
            "but the H2 day-bootstrap interval crosses zero and broad proxy stress was mixed."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "meta_gate_cache.npz",
        valid_index_ns=index.astype("int64").to_numpy(),
        valid_probability=retained_probability.astype("float32"),
        valid_gate=retained_gate,
        valid_candidate=retained_candidate.astype("float32"),
        test_index_ns=test_index.astype("int64").to_numpy(),
        test_probability=test_probability.astype("float32"),
        test_gate=test_gate,
        test_candidate=test_candidate.astype("float32"),
    )
    (artifact_dir / "meta_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--pre-cross-test", default="artifacts_final/lineage_inputs/base_pre_cross.csv"
    )
    parser.add_argument(
        "--current-test",
        default="submissions/archive/blend_best_crossg3_traj5_consensus.csv",
    )
    parser.add_argument(
        "--output",
        default="submissions/blend_best_crossg3_traj_meta25_p55.csv",
    )
    parser.add_argument("--artifact-dir", default="artifacts_final/meta_gate")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run_experiment(
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.pre_cross_test),
        Path(args.current_test),
        Path(args.output),
        Path(args.artifact_dir),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
