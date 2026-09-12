"""Baseline-matched non-crossing residual distribution screen.

The model estimates a conditional distribution of ``truth - incumbent`` and
derives settlement-aware actions around that same incumbent.  This avoids the
baseline mismatch found in the first absolute-distribution experiment.

Validation is fail-closed:

* fit on 2024 Q1 and select one coverage/weight policy on Q2;
* refit on 2024 H1 and confirm the locked policy on H2;
* retain Q1 unchanged and compose a causal April-December OOF surface;
* require component, monthly, issue-block, and complementary 40/60 gates;
* never write a submission candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from experiments.global_capacity_model import make_group_frame
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from experiments.noncrossing_distributional_core import (
    QUANTILE_LEVELS,
    closest_utility_maximizer,
    fit_one_seed,
)
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
COMPONENTS = ("score", "one_minus_nmae", "ficr")
COVERAGES = (0.05, 0.10, 0.20, 0.40)
WEIGHTS = (0.10, 0.20, 0.30, 0.50, 1.00)
DIRECT_ACTION_RATIOS = np.arange(-0.04, 0.0401, 0.01)
Q2_START = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01")
INCUMBENT_PUBLIC_SCORE = 0.6461250914
TARGET_SCORE = 0.65


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def interval_month(index: pd.DatetimeIndex) -> np.ndarray:
    """Month of the generation interval ending at each timestamp."""
    return np.asarray((index - pd.Timedelta(hours=1)).month)


def _metric(
    truth: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, Any]:
    positions = np.asarray(positions)
    sliced_truth = {
        target: np.asarray(truth[target], dtype=float)[positions]
        for target in TARGETS
    }
    sliced_prediction = {
        target: np.asarray(prediction[target], dtype=float)[positions]
        for target in TARGETS
    }
    if all(
        np.any(sliced_truth[target] >= 0.10 * CAPACITY_KWH[target])
        for target in TARGETS
    ):
        return evaluate_competition(sliced_truth, sliced_prediction)

    # A short calendar slice can contain no official evaluation row for one
    # farm. Aggregate the remaining farms with the official equal-group
    # convention instead of crashing or imputing a fictitious zero delta.
    groups = {
        target: evaluate_group(
            sliced_truth[target],
            sliced_prediction[target],
            CAPACITY_KWH[target],
        )
        for target in TARGETS
        if np.any(sliced_truth[target] >= 0.10 * CAPACITY_KWH[target])
    }
    if not groups:
        raise ValueError("No valid evaluation rows in the requested slice.")
    mean_nmae = float(np.mean([result.nmae for result in groups.values()]))
    mean_ficr = float(np.mean([result.ficr for result in groups.values()]))
    return {
        "score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
        "one_minus_nmae": 1.0 - mean_nmae,
        "ficr": mean_ficr,
        "groups": {
            target: result.to_dict()
            for target, result in groups.items()
        },
    }


def metric_delta(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, float]:
    before = _metric(truth, reference, positions)
    after = _metric(truth, candidate, positions)
    return {
        component: float(after[component] - before[component])
        for component in COMPONENTS
    }


def _summarize_delta_matrix(values: np.ndarray) -> dict[str, Any]:
    return {
        component: {
            "mean": float(np.mean(values[:, position])),
            "standard_deviation": float(
                np.std(values[:, position], ddof=1)
            ),
            "positive_fraction": float(
                np.mean(values[:, position] > 0.0)
            ),
            "q05": float(np.quantile(values[:, position], 0.05)),
            "median": float(np.quantile(values[:, position], 0.50)),
            "q95": float(np.quantile(values[:, position], 0.95)),
        }
        for position, component in enumerate(COMPONENTS)
    }


def complementary_subset_stress_all(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    rows: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    stratify_month: bool,
) -> dict[str, Any]:
    """Same-timestamp complementary 40/60 stress over all three groups."""
    rows = np.asarray(rows, dtype=bool)
    available = np.flatnonzero(rows)
    if len(available) < 100:
        raise ValueError("subset stress needs at least 100 timestamps")
    months = interval_month(index)
    strata = (
        [
            available[months[available] == month]
            for month in sorted(set(months[available]))
        ]
        if stratify_month
        else [available]
    )
    rng = np.random.default_rng(seed)
    public = np.empty((repetitions, len(COMPONENTS)), dtype=float)
    private = np.empty_like(public)
    for iteration in range(repetitions):
        selected = np.concatenate(
            [
                rng.choice(
                    stratum,
                    size=max(1, int(round(0.40 * len(stratum)))),
                    replace=False,
                )
                for stratum in strata
            ]
        )
        public_rows = np.zeros(len(index), dtype=bool)
        public_rows[selected] = True
        public_rows &= rows
        private_rows = rows & ~public_rows
        public_delta = metric_delta(
            truth, reference, candidate, public_rows
        )
        private_delta = metric_delta(
            truth, reference, candidate, private_rows
        )
        public[iteration] = [
            public_delta[component] for component in COMPONENTS
        ]
        private[iteration] = [
            private_delta[component] for component in COMPONENTS
        ]
    return {
        "contract": {
            "public_fraction": 0.40,
            "private_fraction": 0.60,
            "complementary_splits": True,
            "same_timestamp_subset_for_all_groups": True,
            "stratified_by_interval_month": stratify_month,
            "sampling_without_replacement": True,
        },
        "repetitions": repetitions,
        "public": _summarize_delta_matrix(public),
        "private": _summarize_delta_matrix(private),
    }


def issue_block_bootstrap_all(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    issues: pd.Series,
    rows: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    issue_values = pd.to_datetime(issues).to_numpy()
    finite = rows & ~pd.isna(issue_values)
    months = interval_month(index)
    position_map: dict[np.datetime64, np.ndarray] = {
        issue: np.flatnonzero(finite & (issue_values == issue))
        for issue in np.unique(issue_values[finite])
    }
    strata: dict[int, np.ndarray] = {}
    for issue, positions in position_map.items():
        month = int(months[positions[0]])
        strata.setdefault(month, [])
        strata[month].append(issue)
    stratum_arrays = {
        month: np.asarray(values)
        for month, values in strata.items()
    }
    rng = np.random.default_rng(seed)
    values = np.empty((repetitions, len(COMPONENTS)), dtype=float)
    for iteration in range(repetitions):
        sampled_positions: list[np.ndarray] = []
        for issue_pool in stratum_arrays.values():
            sampled = rng.choice(
                issue_pool, size=len(issue_pool), replace=True
            )
            sampled_positions.extend(position_map[issue] for issue in sampled)
        positions = np.concatenate(sampled_positions)
        delta = metric_delta(truth, reference, candidate, positions)
        values[iteration] = [
            delta[component] for component in COMPONENTS
        ]
    return {
        "contract": {
            "unit": "complete forecast issue cycle",
            "stratified_by_interval_month": True,
            "sampling_with_replacement": True,
        },
        "repetitions": repetitions,
        "issue_cycles": int(len(position_map)),
        "summary": _summarize_delta_matrix(values),
    }


def _subset_pass(result: dict[str, Any]) -> bool:
    return bool(
        all(
            result[split][component]["q05"] >= 0.0
            for split in ("public", "private")
            for component in COMPONENTS
        )
    )


def _prepare_frames(
    feature_frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    baseline: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    for target_position, target in enumerate(TARGETS):
        frame = make_group_frame(feature_frame, target).reindex(index).copy()
        frame["incumbent_ratio"] = (
            np.asarray(baseline[target], dtype=float) / CAPACITY_KWH[target]
        ).astype("float32")
        for group_position, group in enumerate(TARGETS):
            frame[f"target_is_{group}"] = np.float32(
                target_position == group_position
            )
        frames[target] = frame.astype("float32")
    columns = list(frames[TARGETS[0]].columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise ValueError("pooled residual frames do not share a schema")
    if any(frame.isna().any().any() for frame in frames.values()):
        raise ValueError("pooled residual frames contain missing values")
    return frames, columns


def _stack_features(
    frames: dict[str, pd.DataFrame],
    rows: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=bool)
    return np.concatenate(
        [frames[target].loc[rows].to_numpy(dtype=np.float32) for target in TARGETS]
    )


def _stack_target(
    target: dict[str, np.ndarray],
    rows: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=bool)
    return np.concatenate(
        [np.asarray(target[group], dtype=np.float32)[rows] for group in TARGETS]
    )


def fit_residual_distribution(
    frames: dict[str, pd.DataFrame],
    scaled_residual: dict[str, np.ndarray],
    *,
    train_rows: np.ndarray,
    inner_train_rows: np.ndarray,
    inner_valid_rows: np.ndarray,
    query_rows: np.ndarray,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    label: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    train_rows = np.asarray(train_rows, dtype=bool)
    train_features = _stack_features(frames, train_rows)
    train_target = _stack_target(scaled_residual, train_rows)
    query_features = _stack_features(frames, query_rows)
    inner_train = np.concatenate(
        [np.asarray(inner_train_rows, dtype=bool)[train_rows] for _ in TARGETS]
    )
    inner_valid = np.concatenate(
        [np.asarray(inner_valid_rows, dtype=bool)[train_rows] for _ in TARGETS]
    )
    members: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        prediction, record = fit_one_seed(
            train_features,
            train_target,
            inner_train,
            inner_valid,
            query_features,
            seed=seed,
            hidden=args.hidden,
            dropout=args.dropout,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        members.append(prediction)
        records.append({"seed": seed, **record})
        print(
            f"{label} seed={seed} epoch={record['best_epoch']} "
            f"inner_pinball={record['best_inner_pinball']:.6f}",
            flush=True,
        )
    ensemble = np.mean(members, axis=0)
    if np.any(np.diff(ensemble, axis=1) < -1e-7):
        raise AssertionError("residual ensemble quantiles crossed")
    rows_per_target = int(np.asarray(query_rows, dtype=bool).sum())
    return {
        target: ensemble[
            position * rows_per_target : (position + 1) * rows_per_target
        ]
        for position, target in enumerate(TARGETS)
    }, records


def direct_utility_advantage(
    truth: np.ndarray,
    incumbent: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
    action_ratios: np.ndarray = DIRECT_ACTION_RATIOS,
) -> np.ndarray:
    """Exact row-level official-utility gain for bounded actions."""
    truth = np.asarray(truth, dtype=float)
    incumbent = np.asarray(incumbent, dtype=float)
    action_ratios = np.asarray(action_ratios, dtype=float)
    candidates = np.clip(
        incumbent[:, None] + capacity * action_ratios[None, :],
        0.0,
        capacity,
    )
    eligible = truth >= 0.10 * capacity

    def reward(prediction: np.ndarray) -> np.ndarray:
        error = np.abs(prediction - truth[:, None]) / capacity
        units = np.where(
            error <= 0.06,
            1.0,
            np.where(error <= 0.08, 0.75, 0.0),
        )
        return np.where(
            eligible[:, None],
            -0.5 * error
            + 0.5
            * truth[:, None]
            / max(float(mean_eligible_generation), 1.0)
            * units,
            0.0,
        )

    candidate_reward = reward(candidates)
    incumbent_reward = reward(incumbent[:, None])
    return (candidate_reward - incumbent_reward).astype("float32")


def _make_direct_utility_model(
    seed: int,
    *,
    n_estimators: int,
    args: argparse.Namespace,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=int(n_estimators),
        learning_rate=args.direct_learning_rate,
        num_leaves=args.direct_num_leaves,
        min_child_samples=args.direct_min_child_samples,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=args.direct_jobs,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
    )


def fit_direct_utility_action(
    frames: dict[str, pd.DataFrame],
    truth: dict[str, np.ndarray],
    incumbent: dict[str, np.ndarray],
    *,
    train_rows: np.ndarray,
    inner_train_rows: np.ndarray,
    inner_valid_rows: np.ndarray,
    query_rows: np.ndarray,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    label: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Learn conditional official-utility uplift as a direct value model."""
    train_rows = np.asarray(train_rows, dtype=bool)
    query_rows = np.asarray(query_rows, dtype=bool)
    action_ratios = DIRECT_ACTION_RATIOS
    action_count = len(action_ratios)
    train_base = _stack_features(frames, train_rows)
    query_base = _stack_features(frames, query_rows)
    train_features = np.column_stack(
        [
            np.repeat(train_base, action_count, axis=0),
            np.tile(action_ratios, len(train_base)).astype("float32"),
        ]
    ).astype("float32", copy=False)
    query_features = np.column_stack(
        [
            np.repeat(query_base, action_count, axis=0),
            np.tile(action_ratios, len(query_base)).astype("float32"),
        ]
    ).astype("float32", copy=False)

    utility_targets: list[np.ndarray] = []
    for target in TARGETS:
        capacity = CAPACITY_KWH[target]
        eligible_truth = np.asarray(truth[target], dtype=float)[train_rows]
        eligible_truth = eligible_truth[
            eligible_truth >= 0.10 * capacity
        ]
        utility_targets.append(
            direct_utility_advantage(
                np.asarray(truth[target], dtype=float)[train_rows],
                np.asarray(incumbent[target], dtype=float)[train_rows],
                capacity=capacity,
                mean_eligible_generation=float(np.mean(eligible_truth)),
                action_ratios=action_ratios,
            )
        )
    train_target = np.concatenate(utility_targets, axis=0).reshape(-1)
    stacked_inner_train = np.concatenate(
        [np.asarray(inner_train_rows, dtype=bool)[train_rows] for _ in TARGETS]
    )
    stacked_inner_valid = np.concatenate(
        [np.asarray(inner_valid_rows, dtype=bool)[train_rows] for _ in TARGETS]
    )
    inner_train = np.repeat(stacked_inner_train, action_count)
    inner_valid = np.repeat(stacked_inner_valid, action_count)

    predictions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        selector = _make_direct_utility_model(
            seed,
            n_estimators=args.direct_maximum_estimators,
            args=args,
        )
        selector.fit(
            train_features[inner_train],
            train_target[inner_train],
            eval_set=[
                (
                    train_features[inner_valid],
                    train_target[inner_valid],
                )
            ],
            callbacks=[
                lgb.early_stopping(
                    args.direct_patience, verbose=False
                ),
                lgb.log_evaluation(0),
            ],
        )
        best_iteration = max(1, int(selector.best_iteration_))
        refit = _make_direct_utility_model(
            seed,
            n_estimators=best_iteration,
            args=args,
        )
        refit.fit(
            train_features,
            train_target,
            callbacks=[lgb.log_evaluation(0)],
        )
        prediction = refit.predict(query_features).reshape(
            len(query_base), action_count
        )
        inner_prediction = selector.predict(
            train_features[inner_valid],
            num_iteration=best_iteration,
        )
        inner_mae = float(
            np.mean(
                np.abs(
                    inner_prediction - train_target[inner_valid]
                )
            )
        )
        predictions.append(prediction)
        records.append(
            {
                "seed": seed,
                "best_iteration": best_iteration,
                "inner_utility_mae": inner_mae,
            }
        )
        print(
            f"{label} seed={seed} iteration={best_iteration} "
            f"inner_utility_mae={inner_mae:.6f}",
            flush=True,
        )
    ensemble = np.mean(predictions, axis=0)
    rows_per_target = int(query_rows.sum())
    actions: dict[str, np.ndarray] = {}
    advantages: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for position, target in enumerate(TARGETS):
        start = position * rows_per_target
        stop = (position + 1) * rows_per_target
        utility = ensemble[start:stop]
        base = np.asarray(incumbent[target], dtype=float)[query_rows]
        candidates = np.clip(
            base[:, None]
            + CAPACITY_KWH[target] * action_ratios[None, :],
            0.0,
            CAPACITY_KWH[target],
        )
        selected = closest_utility_maximizer(
            utility, candidates, base
        )
        row = np.arange(rows_per_target)
        actions[target] = candidates[row, selected]
        advantages[target] = np.maximum(utility[row, selected], 0.0)
        movement = (
            np.abs(actions[target] - base) / CAPACITY_KWH[target]
        )
        diagnostics[target] = {
            "nonzero_action_fraction": float(
                np.mean(movement > 1e-10)
            ),
            "mean_action_ratio": float(np.mean(movement)),
            "p95_action_ratio": float(np.quantile(movement, 0.95)),
            "maximum_action_ratio": float(np.max(movement)),
            "positive_advantage_fraction": float(
                np.mean(advantages[target] > 0.0)
            ),
        }
    return actions, advantages, records, diagnostics


def residual_bayes_action(
    quantiles: np.ndarray,
    incumbent: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    quantiles = np.asarray(quantiles, dtype=float)
    incumbent = np.asarray(incumbent, dtype=float)
    residual_ratio = 2.0 * quantiles - 1.0
    scenarios = np.clip(
        incumbent[:, None] + residual_ratio * capacity,
        0.0,
        capacity,
    )
    offsets = np.arange(-0.04, 0.0401, 0.01) * capacity
    candidates = np.clip(
        incumbent[:, None] + offsets[None, :],
        0.0,
        capacity,
    )
    eligible = scenarios >= 0.10 * capacity
    error = (
        np.abs(candidates[:, :, None] - scenarios[:, None, :]) / capacity
    )
    units = np.where(error <= 0.06, 1.0, np.where(error <= 0.08, 0.75, 0.0))
    utility = np.where(
        eligible[:, None, :],
        -0.5 * error
        + 0.5
        * scenarios[:, None, :]
        / max(float(mean_eligible_generation), 1.0)
        * units,
        0.0,
    ).mean(axis=2)
    selected = closest_utility_maximizer(
        utility, candidates, incumbent
    )
    action = candidates[np.arange(len(candidates)), selected]
    zero_position = int(np.argmin(np.abs(offsets)))
    advantage = (
        utility[np.arange(len(candidates)), selected]
        - utility[:, zero_position]
    )
    movement = np.abs(action - incumbent) / capacity
    return action, advantage, {
        "nonzero_action_fraction": float(np.mean(movement > 1e-10)),
        "mean_action_ratio": float(np.mean(movement)),
        "p95_action_ratio": float(np.quantile(movement, 0.95)),
        "maximum_action_ratio": float(np.max(movement)),
        "positive_advantage_fraction": float(np.mean(advantage > 0.0)),
    }


def apply_policy(
    incumbent: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    advantage: dict[str, np.ndarray],
    *,
    coverage: float,
    weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidate: dict[str, np.ndarray] = {}
    gates: dict[str, np.ndarray] = {}
    for target in TARGETS:
        base = np.asarray(incumbent[target], dtype=float)
        benefit = np.asarray(advantage[target], dtype=float)
        threshold = float(np.quantile(benefit, 1.0 - coverage))
        gate = (benefit > 0.0) & (benefit >= threshold)
        candidate[target] = np.clip(
            base
            + gate
            * float(weight)
            * (np.asarray(action[target], dtype=float) - base),
            0.0,
            CAPACITY_KWH[target],
        )
        gates[target] = gate
    return candidate, gates


def movement_summary(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, float]:
    movement = np.concatenate(
        [
            np.abs(
                np.asarray(candidate[target], dtype=float)
                - np.asarray(reference[target], dtype=float)
            )
            / CAPACITY_KWH[target]
            for target in TARGETS
        ]
    )
    return {
        "changed_fraction": float(np.mean(movement > 1e-10)),
        "mean_ratio": float(np.mean(movement)),
        "p95_ratio": float(np.quantile(movement, 0.95)),
        "maximum_ratio": float(np.max(movement)),
    }


def _period_candidate(
    full_reference: dict[str, np.ndarray],
    period_candidate: dict[str, np.ndarray],
    period_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {
        target: np.asarray(values, dtype=float).copy()
        for target, values in full_reference.items()
    }
    for target in TARGETS:
        output[target][period_rows] = period_candidate[target]
    return output


def _selection_record(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    q2_rows: np.ndarray,
    *,
    coverage: float,
    weight: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    months = interval_month(index)
    full = metric_delta(truth, reference, candidate, q2_rows)
    monthly = {
        str(month): metric_delta(
            truth,
            reference,
            candidate,
            q2_rows & (months == month),
        )
        for month in sorted(set(months[q2_rows]))
    }
    movement = movement_summary(reference, candidate)
    preeligible = bool(
        all(full[component] > 0.0 for component in COMPONENTS)
        and all(row["score"] >= 0.0 for row in monthly.values())
        and movement["changed_fraction"] > 0.0
        and movement["p95_ratio"] <= 0.04
    )
    iid = None
    stratified = None
    eligible = False
    if preeligible:
        iid = complementary_subset_stress_all(
            truth,
            reference,
            candidate,
            index,
            q2_rows,
            repetitions=repetitions,
            seed=seed,
            stratify_month=False,
        )
        stratified = complementary_subset_stress_all(
            truth,
            reference,
            candidate,
            index,
            q2_rows,
            repetitions=repetitions,
            seed=seed + 1,
            stratify_month=True,
        )
        eligible = _subset_pass(iid) and _subset_pass(stratified)
    return {
        "coverage": coverage,
        "weight": weight,
        "full_delta": full,
        "monthly_deltas": monthly,
        "movement": movement,
        "preeligible": preeligible,
        "iid_subset_stress": iid,
        "month_stratified_subset_stress": stratified,
        "eligible": bool(eligible),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.torch_threads)
    feature_frame = pd.read_pickle(_rooted(args.feature_train))
    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    reference = {
        target: baselines[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    frames, columns = _prepare_frames(feature_frame, index, reference)
    scaled_residual = {
        target: np.clip(
            0.5
            * (
                (truth[target] - reference[target])
                / CAPACITY_KWH[target]
                + 1.0
            ),
            0.0,
            1.0,
        ).astype("float32")
        for target in TARGETS
    }
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    q1 = np.asarray(index < Q2_START)
    q2 = np.asarray((index >= Q2_START) & (index < H2_START))
    h1 = np.asarray(index < H2_START)
    h2 = np.asarray(index >= H2_START)
    q1_inner_train = np.asarray(index < pd.Timestamp("2024-03-01"))
    q1_inner_valid = np.asarray(
        (index >= pd.Timestamp("2024-03-01")) & q1
    )
    h1_inner_train = np.asarray(index < pd.Timestamp("2024-06-01"))
    h1_inner_valid = np.asarray(
        (index >= pd.Timestamp("2024-06-01")) & h1
    )

    q2_reference = {
        target: reference[target][q2] for target in TARGETS
    }
    if args.estimator == "direct_utility":
        (
            q2_action,
            q2_advantage,
            fit_q1,
            q2_action_diagnostics,
        ) = fit_direct_utility_action(
            frames,
            truth,
            reference,
            train_rows=q1,
            inner_train_rows=q1_inner_train,
            inner_valid_rows=q1_inner_valid,
            query_rows=q2,
            seeds=seeds,
            args=args,
            label="Q1->Q2 direct utility",
        )
    else:
        quantiles_q2, fit_q1 = fit_residual_distribution(
            frames,
            scaled_residual,
            train_rows=q1,
            inner_train_rows=q1_inner_train,
            inner_valid_rows=q1_inner_valid,
            query_rows=q2,
            seeds=seeds,
            args=args,
            label="Q1->Q2",
        )
        q2_action = {}
        q2_advantage = {}
        q2_action_diagnostics = {}
        for target in TARGETS:
            capacity = CAPACITY_KWH[target]
            eligible_truth = truth[target][q1]
            eligible_truth = eligible_truth[
                eligible_truth >= 0.10 * capacity
            ]
            (
                q2_action[target],
                q2_advantage[target],
                q2_action_diagnostics[target],
            ) = residual_bayes_action(
                quantiles_q2[target],
                q2_reference[target],
                capacity=capacity,
                mean_eligible_generation=float(np.mean(eligible_truth)),
            )

    selection_records: list[dict[str, Any]] = []
    selection_candidates: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for coverage in COVERAGES:
        for weight in WEIGHTS:
            period_candidate, _ = apply_policy(
                q2_reference,
                q2_action,
                q2_advantage,
                coverage=coverage,
                weight=weight,
            )
            candidate = _period_candidate(reference, period_candidate, q2)
            selection_candidates[(coverage, weight)] = candidate
            selection_records.append(
                _selection_record(
                    truth,
                    reference,
                    candidate,
                    index,
                    q2,
                    coverage=coverage,
                    weight=weight,
                    repetitions=args.selection_repetitions,
                    seed=20260729
                    + 100 * len(selection_records),
                )
            )
    eligible_records = [
        record for record in selection_records if record["eligible"]
    ]
    selected = (
        max(
            eligible_records,
            key=lambda record: (
                record["full_delta"]["score"],
                min(record["full_delta"].values()),
                -record["coverage"],
                -record["weight"],
            ),
        )
        if eligible_records
        else None
    )

    fit_h1: list[dict[str, Any]] = []
    confirmation: dict[str, Any] | None = None
    cache: dict[str, np.ndarray] = {
        "index_ns": index.astype("int64").to_numpy(),
    }
    if selected is not None:
        h2_reference = {
            target: reference[target][h2] for target in TARGETS
        }
        if args.estimator == "direct_utility":
            (
                h2_action,
                h2_advantage,
                fit_h1,
                h2_action_diagnostics,
            ) = fit_direct_utility_action(
                frames,
                truth,
                reference,
                train_rows=h1,
                inner_train_rows=h1_inner_train,
                inner_valid_rows=h1_inner_valid,
                query_rows=h2,
                seeds=seeds,
                args=args,
                label="H1->H2 direct utility",
            )
        else:
            quantiles_h2, fit_h1 = fit_residual_distribution(
                frames,
                scaled_residual,
                train_rows=h1,
                inner_train_rows=h1_inner_train,
                inner_valid_rows=h1_inner_valid,
                query_rows=h2,
                seeds=seeds,
                args=args,
                label="H1->H2",
            )
            h2_action = {}
            h2_advantage = {}
            h2_action_diagnostics = {}
            for target in TARGETS:
                capacity = CAPACITY_KWH[target]
                eligible_truth = truth[target][h1]
                eligible_truth = eligible_truth[
                    eligible_truth >= 0.10 * capacity
                ]
                (
                    h2_action[target],
                    h2_advantage[target],
                    h2_action_diagnostics[target],
                ) = residual_bayes_action(
                    quantiles_h2[target],
                    h2_reference[target],
                    capacity=capacity,
                    mean_eligible_generation=float(
                        np.mean(eligible_truth)
                    ),
                )
        h2_period_candidate, h2_gates = apply_policy(
            h2_reference,
            h2_action,
            h2_advantage,
            coverage=float(selected["coverage"]),
            weight=float(selected["weight"]),
        )
        q2_candidate = selection_candidates[
            (float(selected["coverage"]), float(selected["weight"]))
        ]
        causal_candidate = {
            target: q2_candidate[target].copy() for target in TARGETS
        }
        for target in TARGETS:
            causal_candidate[target][h2] = h2_period_candidate[target]
            cache[f"{target}__q2_action"] = q2_action[target].astype(
                "float32"
            )
            cache[f"{target}__q2_advantage"] = q2_advantage[target].astype(
                "float32"
            )
            cache[f"{target}__h2_action"] = h2_action[target].astype(
                "float32"
            )
            cache[f"{target}__h2_advantage"] = h2_advantage[target].astype(
                "float32"
            )
            cache[f"{target}__causal_candidate"] = causal_candidate[
                target
            ].astype("float32")

        months = interval_month(index)
        h2_delta = metric_delta(
            truth, reference, causal_candidate, h2
        )
        full_delta = metric_delta(
            truth,
            reference,
            causal_candidate,
            np.ones(len(index), dtype=bool),
        )
        h2_monthly = {
            str(month): metric_delta(
                truth,
                reference,
                causal_candidate,
                h2 & (months == month),
            )
            for month in sorted(set(months[h2]))
        }
        h2_iid = complementary_subset_stress_all(
            truth,
            reference,
            causal_candidate,
            index,
            h2,
            repetitions=args.confirmation_repetitions,
            seed=20260801,
            stratify_month=False,
        )
        h2_stratified = complementary_subset_stress_all(
            truth,
            reference,
            causal_candidate,
            index,
            h2,
            repetitions=args.confirmation_repetitions,
            seed=20260802,
            stratify_month=True,
        )
        full_iid = complementary_subset_stress_all(
            truth,
            reference,
            causal_candidate,
            index,
            np.ones(len(index), dtype=bool),
            repetitions=args.confirmation_repetitions,
            seed=20260803,
            stratify_month=False,
        )
        full_stratified = complementary_subset_stress_all(
            truth,
            reference,
            causal_candidate,
            index,
            np.ones(len(index), dtype=bool),
            repetitions=args.confirmation_repetitions,
            seed=20260804,
            stratify_month=True,
        )
        issue_bootstrap = issue_block_bootstrap_all(
            truth,
            reference,
            causal_candidate,
            index,
            issues,
            h2,
            repetitions=args.issue_repetitions,
            seed=20260805,
        )
        movement = movement_summary(reference, causal_candidate)
        projected_score = INCUMBENT_PUBLIC_SCORE + full_delta["score"]
        gates = {
            "q2_selection_subset_safe": True,
            "h2_all_components_positive": bool(
                all(h2_delta[component] > 0.0 for component in COMPONENTS)
            ),
            "every_h2_interval_month_score_nonnegative": bool(
                all(
                    delta["score"] >= 0.0
                    for delta in h2_monthly.values()
                )
            ),
            "h2_iid_40_60_q05_nonnegative": _subset_pass(h2_iid),
            "h2_stratified_40_60_q05_nonnegative": _subset_pass(
                h2_stratified
            ),
            "full_iid_40_60_q05_nonnegative": _subset_pass(full_iid),
            "full_stratified_40_60_q05_nonnegative": _subset_pass(
                full_stratified
            ),
            "h2_issue_q05_all_components_nonnegative": bool(
                all(
                    issue_bootstrap["summary"][component]["q05"] >= 0.0
                    for component in COMPONENTS
                )
            ),
            "p95_movement_at_most_4pct": bool(
                movement["p95_ratio"] <= 0.04
            ),
        }
        confirmation = {
            "action_diagnostics": h2_action_diagnostics,
            "changed_rows_by_target": {
                target: int(h2_gates[target].sum()) for target in TARGETS
            },
            "h2_delta": h2_delta,
            "full_causal_oof_delta": full_delta,
            "h2_monthly_deltas": h2_monthly,
            "h2_iid_subset_stress": h2_iid,
            "h2_month_stratified_subset_stress": h2_stratified,
            "full_iid_subset_stress": full_iid,
            "full_month_stratified_subset_stress": full_stratified,
            "h2_issue_block_bootstrap": issue_bootstrap,
            "movement": movement,
            "projected_public_score_if_local_full_delta_transfers": (
                projected_score
            ),
            "projected_gap_to_065": TARGET_SCORE - projected_score,
            "gates": gates,
            "robust_qualified": bool(all(gates.values())),
            "target_065_reached_locally": bool(
                projected_score >= TARGET_SCORE
            ),
        }

    cache_path = _rooted(args.output_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    qualified = bool(
        confirmation is not None
        and confirmation["robust_qualified"]
        and confirmation["target_065_reached_locally"]
    )
    report = {
        "schema_version": "incumbent_residual_decision.v2",
        "family": (
            "incumbent_matched_direct_utility"
            if args.estimator == "direct_utility"
            else "incumbent_matched_noncrossing_residual_distribution"
        ),
        "method": (
            (
                "pooled three-group direct conditional value model for the "
                "exact official-utility gain of bounded incumbent actions"
            )
            if args.estimator == "direct_utility"
            else (
                "pooled three-group non-crossing residual quantiles "
                "conditioned on the exact incumbent; settlement-aware "
                "action around the same incumbent"
            )
        ),
        "validation_contract": {
            "development": "2024 Q1 fit -> Q2 policy selection",
            "confirmation": "2024 H1 fit -> locked H2",
            "causal_surface": (
                "Q1 unchanged, Q2 development prediction, H2 locked "
                "confirmation prediction"
            ),
            "baseline_matched": True,
            "public_score_used_for_model_or_policy_selection": False,
            "2025_labels_used": False,
            "submission_side_effect": False,
            "h2_previously_exposed_to_other_model_families": True,
        },
        "configuration": {
            "estimator": args.estimator,
            "targets": TARGETS,
            "feature_count": len(columns),
            "quantile_levels": QUANTILE_LEVELS.tolist(),
            "direct_action_ratios": DIRECT_ACTION_RATIOS.tolist(),
            "coverages": COVERAGES,
            "weights": WEIGHTS,
            "seeds": seeds,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "maximum_epochs": args.maximum_epochs,
            "patience": args.patience,
            "direct_maximum_estimators": (
                args.direct_maximum_estimators
            ),
            "direct_patience": args.direct_patience,
            "direct_num_leaves": args.direct_num_leaves,
            "direct_min_child_samples": (
                args.direct_min_child_samples
            ),
        },
        "residual_target": {
            "definition": "(truth - incumbent) / capacity",
            "support": [-1.0, 1.0],
            "scaled_training_support": [0.0, 1.0],
            "clipped_fraction": float(
                np.mean(
                    np.concatenate(
                        [
                            (
                                np.abs(
                                    (truth[target] - reference[target])
                                    / CAPACITY_KWH[target]
                                )
                                >= 1.0
                            )
                            for target in TARGETS
                        ]
                    )
                )
            ),
        },
        "fit_q1": fit_q1,
        "selection_q2": {
            "action_diagnostics": q2_action_diagnostics,
            "records": selection_records,
            "selected": selected,
        },
        "fit_h1": fit_h1,
        "confirmation_h2": confirmation,
        "artifact_cache": cache_path.relative_to(ROOT).as_posix(),
        "promotion": {
            "selection_eligible": selected is not None,
            "confirmation_opened": confirmation is not None,
            "robust_qualified": bool(
                confirmation is not None
                and confirmation["robust_qualified"]
            ),
            "target_065_reached_locally": bool(
                confirmation is not None
                and confirmation["target_065_reached_locally"]
            ),
            "qualified_for_065": qualified,
            "candidate_written": False,
            "decision": (
                "research_pass_no_candidate_writer"
                if qualified
                else "rejected_fail_closed"
            ),
        },
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-train",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--estimator",
        choices=("noncrossing_quantile", "direct_utility"),
        default="noncrossing_quantile",
    )
    parser.add_argument("--seeds", default="17,41")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--direct-maximum-estimators", type=int, default=800
    )
    parser.add_argument("--direct-patience", type=int, default=50)
    parser.add_argument(
        "--direct-learning-rate", type=float, default=0.03
    )
    parser.add_argument("--direct-num-leaves", type=int, default=31)
    parser.add_argument(
        "--direct-min-child-samples", type=int, default=80
    )
    parser.add_argument("--direct-jobs", type=int, default=4)
    parser.add_argument("--selection-repetitions", type=int, default=500)
    parser.add_argument(
        "--confirmation-repetitions", type=int, default=5_000
    )
    parser.add_argument("--issue-repetitions", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "incumbent_residual_noncrossing_20260729.json"
        ),
    )
    parser.add_argument(
        "--output-cache",
        default=(
            "artifacts_final/lineage/"
            "incumbent_residual_noncrossing_20260729.npz"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected": report["selection_q2"]["selected"],
                "confirmation": (
                    None
                    if report["confirmation_h2"] is None
                    else {
                        "h2_delta": report["confirmation_h2"]["h2_delta"],
                        "full_delta": report["confirmation_h2"][
                            "full_causal_oof_delta"
                        ],
                        "projected_public_score": report["confirmation_h2"][
                            "projected_public_score_if_local_full_delta_transfers"
                        ],
                        "gates": report["confirmation_h2"]["gates"],
                    }
                ),
                "promotion": report["promotion"],
                "output": str(_rooted(args.output).relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
