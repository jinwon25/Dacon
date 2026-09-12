"""Expanding-window multi-NWP residual expert for the frozen public incumbent.

The experiment uses only operational forecasts available before each BARAM
cutoff.  Hyperparameters are selected on 2024 Q2 after training through Q1.
They are then frozen for two expanding confirmations:

* train through Q2 -> predict Q3
* train through Q3 -> predict Q4

Only targets that improve score, 1-NMAE, and FICR in both confirmation folds,
all seed/fold combinations, the H2 aggregate, and issue-cycle bootstrap are
eligible for deployment.  Production fits the frozen architecture on the
available 2024 history and predicts 2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)
from experiments.kma_year_forward_quantile_blend import (
    apply_bounded_blend,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
ALPHAS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
SEEDS = (42, 202, 2026)
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context_bundle(
    paths: tuple[Path, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join causal model features and keep the primary issue timestamp."""
    if not paths:
        raise ValueError("at least one context source is required")
    blocks: list[pd.DataFrame] = []
    issue: pd.Series | None = None
    for position, path in enumerate(paths, start=1):
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"],
        )
        if frame["forecast_kst_dtm"].duplicated().any():
            raise ValueError(f"context has duplicate times: {path}")
        frame = frame.set_index("forecast_kst_dtm").sort_index()
        columns = [
            column for column in frame if column.startswith("kma_um_ctx_")
        ]
        if not columns:
            raise ValueError(f"context contains no model features: {path}")
        block = frame[columns].apply(pd.to_numeric, errors="coerce")
        block = block.add_prefix(f"source{position}__")
        if block.isna().any().any():
            raise ValueError(f"context features are incomplete: {path}")
        blocks.append(block)
        if issue is None:
            issue = frame["data_available_kst_dtm"].copy()
    features = pd.concat(blocks, axis=1, join="inner")
    if features.empty:
        raise ValueError("context sources have no common timestamps")
    index = pd.DatetimeIndex(features.index)
    day = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float)
    features["doy_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    features["doy_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    if not features.columns.is_unique:
        raise ValueError("context feature names collide")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("combined context features are non-finite")
    assert issue is not None
    return features.astype("float32"), issue.reindex(features.index)


def make_pooled_design(
    features: pd.DataFrame,
    baselines: dict[str, pd.Series],
    targets: tuple[str, ...] = TARGETS,
) -> tuple[pd.DataFrame, dict[str, slice]]:
    """Stack groups with their frozen baseline and deterministic one-hot IDs."""
    blocks: list[pd.DataFrame] = []
    slices: dict[str, slice] = {}
    start = 0
    for target in targets:
        block = features.reset_index(drop=True).copy()
        baseline = baselines[target].reindex(features.index)
        if baseline.isna().any():
            raise ValueError(f"baseline is incomplete for {target}")
        block["frozen_baseline_ratio"] = (
            baseline.to_numpy(dtype=float) / CAPACITY_KWH[target]
        )
        for supported in targets:
            block[f"pooled_group__{supported}"] = float(target == supported)
        stop = start + len(block)
        slices[target] = slice(start, stop)
        blocks.append(block)
        start = stop
    pooled = pd.concat(blocks, axis=0, ignore_index=True)
    if not np.isfinite(pooled.to_numpy(dtype=float)).all():
        raise ValueError("pooled design is non-finite")
    return pooled.astype("float32"), slices


def make_pooled_target(
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
    targets: tuple[str, ...] = TARGETS,
) -> np.ndarray:
    values = [
        labels[target].reindex(index).to_numpy(dtype=float)
        / CAPACITY_KWH[target]
        for target in targets
    ]
    return np.concatenate(values)


def make_model(
    alpha: float,
    seed: int,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_pooled_expert(
    train_features: pd.DataFrame,
    train_baselines: dict[str, pd.Series],
    labels: pd.DataFrame,
    query_features: pd.DataFrame,
    query_baselines: dict[str, pd.Series],
    *,
    alpha: float,
    seed: int,
    n_estimators: int,
) -> dict[str, np.ndarray]:
    train_design, _ = make_pooled_design(train_features, train_baselines)
    query_design, slices = make_pooled_design(query_features, query_baselines)
    target = make_pooled_target(
        labels,
        pd.DatetimeIndex(train_features.index),
    )
    observed = np.isfinite(target)
    if int(observed.sum()) < 2_000:
        raise ValueError("pooled expanding model has insufficient labels")
    model = make_model(alpha, seed, n_estimators)
    model.fit(
        train_design.loc[observed],
        target[observed],
        callbacks=[lgb.log_evaluation(0)],
    )
    normalized = np.clip(
        np.asarray(model.predict(query_design), dtype=float),
        0.0,
        1.0,
    )
    return {
        target_name: normalized[slices[target_name]]
        * CAPACITY_KWH[target_name]
        for target_name in TARGETS
    }


def fit_independent_source_ensemble(
    train_source_features: tuple[pd.DataFrame, ...],
    train_baselines: dict[str, pd.Series],
    labels: pd.DataFrame,
    query_source_features: tuple[pd.DataFrame, ...],
    query_baselines: dict[str, pd.Series],
    *,
    alpha: float,
    seed: int,
    n_estimators: int,
    aggregation: str = "median",
) -> dict[str, np.ndarray]:
    """Fit one pooled expert per NWP source before robust aggregation.

    Keeping the source learners independent prevents one provider's scale or
    feature count from dominating a single concatenated tree ensemble.  The
    median is the preregistered robust default; the mean is retained only as a
    deterministic diagnostic option.
    """
    if not train_source_features:
        raise ValueError("at least one independent source is required")
    if len(train_source_features) != len(query_source_features):
        raise ValueError("train and query source counts differ")
    if aggregation not in {"median", "mean"}:
        raise ValueError(f"unsupported source aggregation: {aggregation}")

    train_index = train_source_features[0].index
    query_index = query_source_features[0].index
    if any(not frame.index.equals(train_index) for frame in train_source_features):
        raise ValueError("independent train-source indexes differ")
    if any(not frame.index.equals(query_index) for frame in query_source_features):
        raise ValueError("independent query-source indexes differ")

    members = [
        fit_pooled_expert(
            train_features,
            train_baselines,
            labels,
            query_features,
            query_baselines,
            alpha=alpha,
            seed=seed + 10_000 * source_position,
            n_estimators=n_estimators,
        )
        for source_position, (train_features, query_features) in enumerate(
            zip(train_source_features, query_source_features),
            start=1,
        )
    ]
    reducer = np.median if aggregation == "median" else np.mean
    return {
        target: np.asarray(
            reducer(
                np.stack([member[target] for member in members], axis=0),
                axis=0,
            ),
            dtype=float,
        )
        for target in TARGETS
    }


def fit_configured_expert(
    train_features: pd.DataFrame,
    train_source_features: tuple[pd.DataFrame, ...],
    train_baselines: dict[str, pd.Series],
    labels: pd.DataFrame,
    query_features: pd.DataFrame,
    query_source_features: tuple[pd.DataFrame, ...],
    query_baselines: dict[str, pd.Series],
    *,
    alpha: float,
    seed: int,
    n_estimators: int,
    source_aggregation: str,
) -> dict[str, np.ndarray]:
    if source_aggregation == "joint":
        return fit_pooled_expert(
            train_features,
            train_baselines,
            labels,
            query_features,
            query_baselines,
            alpha=alpha,
            seed=seed,
            n_estimators=n_estimators,
        )
    return fit_independent_source_ensemble(
        train_source_features,
        train_baselines,
        labels,
        query_source_features,
        query_baselines,
        alpha=alpha,
        seed=seed,
        n_estimators=n_estimators,
        aggregation=source_aggregation,
    )


def select_spec(
    truth: np.ndarray,
    reference: np.ndarray,
    experts_by_alpha: dict[float, np.ndarray],
    capacity: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Select alpha and weight on one preregistered development fold."""
    rows: list[dict[str, Any]] = []
    mask = np.ones(len(truth), dtype=bool)
    for alpha, expert in experts_by_alpha.items():
        for weight in WEIGHTS:
            candidate = apply_bounded_blend(
                reference,
                expert,
                weight=weight,
                capacity=capacity,
            )
            delta = metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                mask,
            )
            rows.append(
                {
                    "alpha": float(alpha),
                    "weight": float(weight),
                    "delta": delta,
                    "eligible": bool(min(delta.values()) > 0.0),
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        selected = max(
            rows,
            key=lambda row: (
                row["delta"]["score"],
                min(row["delta"].values()),
                -row["weight"],
            ),
        )
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row["delta"]["score"],
                min(row["delta"].values()),
                -row["weight"],
            ),
        )
    return {
        "alpha": float(selected["alpha"]),
        "weight": float(selected["weight"]),
        "selection_eligible": bool(selected["eligible"]),
    }, rows


def _series(values: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index)


def load_frozen_validation_baselines(
    primary_cache_path: Path,
    residual_cache_path: Path,
    group3_cache_path: Path,
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    pd.DatetimeIndex,
    pd.Series,
]:
    primary = np.load(primary_cache_path, allow_pickle=False)
    residual = np.load(residual_cache_path, allow_pickle=False)
    group3 = np.load(group3_cache_path, allow_pickle=False)
    index = pd.DatetimeIndex(pd.to_datetime(group3["index_ns"]))
    baselines: dict[str, pd.Series] = {}
    truths: dict[str, pd.Series] = {}
    for target in ("kpx_group_1", "kpx_group_2"):
        prefix = f"{target}__"
        target_index = pd.DatetimeIndex(
            pd.to_datetime(primary[f"{prefix}index_ns"])
        )
        if not target_index.equals(index):
            raise ValueError(f"validation indexes differ for {target}")
        truths[target] = _series(primary[f"{prefix}truth"], index)
    baselines["kpx_group_1"] = _series(
        apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=0.125,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        ),
        index,
    )
    baselines["kpx_group_2"] = _series(
        primary["kpx_group_2__candidate"],
        index,
    )
    baselines["kpx_group_3"] = _series(group3["rolling_candidate"], index)
    truths["kpx_group_3"] = _series(group3["truth"], index)
    issues = pd.Series(
        pd.to_datetime(group3["issue_ns"]),
        index=index,
    )
    return baselines, truths, index, issues


def _folds(index: pd.DatetimeIndex) -> tuple[dict[str, Any], ...]:
    definitions = (
        (
            "selection_q2",
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-07-01"),
        ),
        (
            "confirmation_q3",
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2024-10-01"),
        ),
        (
            "confirmation_q4",
            pd.Timestamp("2024-10-01"),
            pd.Timestamp("2024-10-01"),
            pd.Timestamp("2025-01-01"),
        ),
    )
    return tuple(
        {
            "name": name,
            "train": np.asarray(index < train_end),
            "query": np.asarray((index >= query_start) & (index < query_end)),
            "train_end": train_end,
            "query_start": query_start,
            "query_end": query_end,
        }
        for name, train_end, query_start, query_end in definitions
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_2024 = tuple(_rooted(value) for value in args.context_2024)
    context_2025 = tuple(_rooted(value) for value in args.context_2025)
    manifests_2024 = tuple(_rooted(value) for value in args.manifest_2024)
    manifests_2025 = tuple(_rooted(value) for value in args.manifest_2025)
    if not (
        len(context_2024)
        == len(context_2025)
        == len(manifests_2024)
        == len(manifests_2025)
    ):
        raise ValueError("context and manifest source counts differ")
    manifest_validation = {
        f"2024_source_{position}": validate_external_data_manifest(
            path,
            ROOT,
        )
        for position, path in enumerate(manifests_2024, start=1)
    }
    manifest_validation.update(
        {
            f"2025_source_{position}": validate_external_data_manifest(
                path,
                ROOT,
            )
            for position, path in enumerate(manifests_2025, start=1)
        }
    )
    features_2024, context_issue_2024 = load_context_bundle(context_2024)
    features_2025, _ = load_context_bundle(context_2025)
    source_features_2024 = tuple(
        load_context_bundle((path,))[0] for path in context_2024
    )
    source_features_2025 = tuple(
        load_context_bundle((path,))[0] for path in context_2025
    )
    baselines_2024, truths_2024, _, cache_issues = (
        load_frozen_validation_baselines(
            _rooted(args.primary_cache),
            _rooted(args.residual_cache),
            _rooted(args.group3_cache),
        )
    )
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    common_2024 = features_2024.index
    for target in TARGETS:
        common_2024 = common_2024.intersection(baselines_2024[target].index)
        common_2024 = common_2024.intersection(truths_2024[target].index)
    common_2024 = pd.DatetimeIndex(common_2024).sort_values()
    features_2024 = features_2024.reindex(common_2024)
    source_features_2024 = tuple(
        frame.reindex(common_2024) for frame in source_features_2024
    )
    baselines_2024 = {
        target: values.reindex(common_2024)
        for target, values in baselines_2024.items()
    }
    truths_2024 = {
        target: values.reindex(common_2024)
        for target, values in truths_2024.items()
    }
    issues_2024 = context_issue_2024.reindex(common_2024)
    cache_issue_common = cache_issues.reindex(common_2024)
    if cache_issue_common.notna().all():
        # The primary KMA context is the first source.  Its recorded safe
        # availability must agree with the exact OOF lineage when present.
        comparable = pd.to_datetime(issues_2024)
        if not comparable.equals(pd.to_datetime(cache_issue_common)):
            raise ValueError("context and exact OOF issue timestamps differ")

    production_frame = pd.read_csv(
        _rooted(args.active_candidate),
        encoding="utf-8-sig",
        parse_dates=["forecast_kst_dtm"],
    )
    if production_frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("active candidate contains duplicate timestamps")
    production_index = pd.DatetimeIndex(
        production_frame["forecast_kst_dtm"]
    )
    production_baselines = {
        target: pd.Series(
            production_frame[target].to_numpy(dtype=float),
            index=production_index,
        )
        for target in TARGETS
    }
    if not features_2025.index.equals(production_index):
        features_2025 = features_2025.reindex(production_index)
    source_features_2025 = tuple(
        frame.reindex(production_index) for frame in source_features_2025
    )
    if features_2025.isna().any().any():
        raise ValueError("2025 contexts do not cover the submission index")
    if any(frame.isna().any().any() for frame in source_features_2025):
        raise ValueError("2025 independent contexts do not cover the submission index")

    folds = _folds(common_2024)
    selection = folds[0]
    selection_predictions: dict[float, dict[str, np.ndarray]] = {}
    for alpha in ALPHAS:
        selection_predictions[alpha] = fit_configured_expert(
            features_2024.loc[selection["train"]],
            tuple(
                frame.loc[selection["train"]]
                for frame in source_features_2024
            ),
            {
                target: values.loc[selection["train"]]
                for target, values in baselines_2024.items()
            },
            labels,
            features_2024.loc[selection["query"]],
            tuple(
                frame.loc[selection["query"]]
                for frame in source_features_2024
            ),
            {
                target: values.loc[selection["query"]]
                for target, values in baselines_2024.items()
            },
            alpha=alpha,
            seed=13,
            n_estimators=args.n_estimators,
            source_aggregation=args.source_aggregation,
        )

    selected_specs: dict[str, dict[str, float]] = {}
    selection_records: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        query = selection["query"]
        spec, records = select_spec(
            truths_2024[target].loc[query].to_numpy(dtype=float),
            baselines_2024[target].loc[query].to_numpy(dtype=float),
            {
                alpha: prediction[target]
                for alpha, prediction in selection_predictions.items()
            },
            CAPACITY_KWH[target],
        )
        selected_specs[target] = spec
        selection_records[target] = records

    fold_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        fold["name"]: {target: [] for target in TARGETS}
        for fold in folds
    }
    # Retain the independently selected seed-13 surface for the development
    # fold only; confirmation folds are refit and checked over all fixed seeds.
    for target in TARGETS:
        alpha = selected_specs[target]["alpha"]
        fold_predictions[selection["name"]][target].append(
            selection_predictions[alpha][target]
        )

    unique_alphas = sorted(
        {float(spec["alpha"]) for spec in selected_specs.values()}
    )
    for fold in folds[1:]:
        by_alpha_seed: dict[tuple[float, int], dict[str, np.ndarray]] = {}
        for alpha in unique_alphas:
            for seed in SEEDS:
                by_alpha_seed[(alpha, seed)] = fit_configured_expert(
                    features_2024.loc[fold["train"]],
                    tuple(
                        frame.loc[fold["train"]]
                        for frame in source_features_2024
                    ),
                    {
                        target: values.loc[fold["train"]]
                        for target, values in baselines_2024.items()
                    },
                    labels,
                    features_2024.loc[fold["query"]],
                    tuple(
                        frame.loc[fold["query"]]
                        for frame in source_features_2024
                    ),
                    {
                        target: values.loc[fold["query"]]
                        for target, values in baselines_2024.items()
                    },
                    alpha=alpha,
                    seed=seed,
                    n_estimators=args.n_estimators,
                    source_aggregation=args.source_aggregation,
                )
        for target in TARGETS:
            alpha = selected_specs[target]["alpha"]
            fold_predictions[fold["name"]][target] = [
                by_alpha_seed[(alpha, seed)][target] for seed in SEEDS
            ]

    validation: dict[str, Any] = {}
    promoted_targets: list[str] = []
    oof_cache: dict[str, np.ndarray] = {}
    expected_macro_delta = 0.0
    for target in TARGETS:
        capacity = CAPACITY_KWH[target]
        weight = float(selected_specs[target]["weight"])
        fold_reports: dict[str, Any] = {}
        confirmation_seed_components: list[float] = []
        oof_index_parts: list[np.ndarray] = []
        oof_truth_parts: list[np.ndarray] = []
        oof_reference_parts: list[np.ndarray] = []
        oof_candidate_parts: list[np.ndarray] = []
        oof_issue_parts: list[np.ndarray] = []
        for fold in folds:
            query = fold["query"]
            reference = baselines_2024[target].loc[query].to_numpy(dtype=float)
            truth = truths_2024[target].loc[query].to_numpy(dtype=float)
            seed_experts = fold_predictions[fold["name"]][target]
            mean_expert = np.mean(seed_experts, axis=0)
            candidate = apply_bounded_blend(
                reference,
                mean_expert,
                weight=weight,
                capacity=capacity,
            )
            mask = np.ones(len(truth), dtype=bool)
            delta = metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                mask,
            )
            seed_deltas = []
            for seed_position, expert in enumerate(seed_experts):
                seed_candidate = apply_bounded_blend(
                    reference,
                    expert,
                    weight=weight,
                    capacity=capacity,
                )
                seed_delta = metric_delta(
                    truth,
                    reference,
                    seed_candidate,
                    capacity,
                    mask,
                )
                seed_deltas.append(
                    {
                        "seed": (
                            13
                            if fold["name"] == "selection_q2"
                            else SEEDS[seed_position]
                        ),
                        "delta": seed_delta,
                    }
                )
                if fold["name"] != "selection_q2":
                    confirmation_seed_components.extend(seed_delta.values())
            fold_reports[fold["name"]] = {
                "train_rows": int(fold["train"].sum()),
                "query_rows": int(query.sum()),
                "delta": delta,
                "seed_deltas": seed_deltas,
            }
            oof_index_parts.append(common_2024[query].view("int64"))
            oof_truth_parts.append(truth)
            oof_reference_parts.append(reference)
            oof_candidate_parts.append(candidate)
            oof_issue_parts.append(
                pd.to_datetime(issues_2024.loc[query])
                .astype("int64")
                .to_numpy()
            )

        oof_index_ns = np.concatenate(oof_index_parts)
        oof_index = pd.DatetimeIndex(pd.to_datetime(oof_index_ns))
        oof_truth = np.concatenate(oof_truth_parts)
        oof_reference = np.concatenate(oof_reference_parts)
        oof_candidate = np.concatenate(oof_candidate_parts)
        oof_issue_ns = np.concatenate(oof_issue_parts)
        h2 = np.asarray(oof_index >= pd.Timestamp("2024-07-01"))
        h2_delta = metric_delta(
            oof_truth,
            oof_reference,
            oof_candidate,
            capacity,
            h2,
        )
        full_delta = metric_delta(
            oof_truth,
            oof_reference,
            oof_candidate,
            capacity,
            np.ones(len(oof_truth), dtype=bool),
        )
        monthly = {
            str(month): metric_delta(
                oof_truth,
                oof_reference,
                oof_candidate,
                capacity,
                h2 & np.asarray(oof_index.month == month),
            )
            for month in range(7, 13)
        }
        positive_h2_months = int(
            sum(value["score"] > 0.0 for value in monthly.values())
        )
        bootstrap = evaluate_blocked_rolling(
            oof_truth,
            oof_reference,
            oof_candidate,
            oof_index,
            pd.DatetimeIndex(pd.to_datetime(oof_issue_ns)),
            h2 & (oof_truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260727,
        )
        gates = {
            "selection_all_components_positive": bool(
                selected_specs[target]["selection_eligible"]
            ),
            "q3_all_components_positive": bool(
                min(
                    fold_reports["confirmation_q3"]["delta"].values()
                )
                > 0.0
            ),
            "q4_all_components_positive": bool(
                min(
                    fold_reports["confirmation_q4"]["delta"].values()
                )
                > 0.0
            ),
            "h2_all_components_positive": bool(min(h2_delta.values()) > 0.0),
            "all_confirmation_seed_components_positive": bool(
                min(confirmation_seed_components) > 0.0
            ),
            "positive_h2_score_months_at_least_five": bool(
                positive_h2_months >= 5
            ),
            "issue_bootstrap_q05_positive": bool(
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "issue_bootstrap_positive_fraction_at_least_95pct": bool(
                bootstrap["issue_block_bootstrap"]["positive_fraction"]
                >= 0.95
            ),
        }
        promoted = bool(all(gates.values()))
        if promoted:
            promoted_targets.append(target)
            expected_macro_delta += float(h2_delta["score"]) / 3.0
        validation[target] = {
            "selected_spec": selected_specs[target],
            "selection_records": selection_records[target],
            "folds": fold_reports,
            "h2_delta": h2_delta,
            "full_oof_delta": full_delta,
            "monthly_h2_deltas": monthly,
            "positive_h2_score_months": positive_h2_months,
            "issue_block_validation": bootstrap,
            "minimum_confirmation_seed_component": float(
                min(confirmation_seed_components)
            ),
            "gates": gates,
            "promotion": "strict" if promoted else "rejected",
        }
        prefix = f"{target}__"
        oof_cache[f"{prefix}index_ns"] = oof_index_ns
        oof_cache[f"{prefix}issue_ns"] = oof_issue_ns
        oof_cache[f"{prefix}truth"] = oof_truth.astype("float32")
        oof_cache[f"{prefix}reference"] = oof_reference.astype("float32")
        oof_cache[f"{prefix}candidate"] = oof_candidate.astype("float32")

    output_path = _rooted(args.output_submission)
    candidate_record: dict[str, Any] | None = None
    if promoted_targets and not args.validation_only:
        production_predictions_by_alpha: dict[
            float, list[dict[str, np.ndarray]]
        ] = {}
        for alpha in sorted(
            {
                selected_specs[target]["alpha"]
                for target in promoted_targets
            }
        ):
            production_predictions_by_alpha[alpha] = [
                fit_configured_expert(
                    features_2024,
                    source_features_2024,
                    baselines_2024,
                    labels,
                    features_2025,
                    source_features_2025,
                    production_baselines,
                    alpha=alpha,
                    seed=seed,
                    n_estimators=args.n_estimators,
                    source_aggregation=args.source_aggregation,
                )
                for seed in SEEDS
            ]
        output = production_frame.copy()
        for target in promoted_targets:
            alpha = selected_specs[target]["alpha"]
            expert = np.mean(
                [
                    member[target]
                    for member in production_predictions_by_alpha[alpha]
                ],
                axis=0,
            )
            output[target] = apply_bounded_blend(
                production_baselines[target].to_numpy(dtype=float),
                expert,
                weight=selected_specs[target]["weight"],
                capacity=CAPACITY_KWH[target],
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(
                f"CandidateValidator rejected output: {audit.errors}"
            )
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "deployed_targets": promoted_targets,
            "candidate_validator": audit.to_dict(),
        }

    cache_path = _rooted(args.output_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **oof_cache)
    report = {
        "family": (
            "multimodel_expanding_pooled_quantile_blend"
            if args.source_aggregation == "joint"
            else "multisource_independent_robust_quantile_ensemble"
        ),
        "method": (
            "frozen Q2 alpha/weight selection; expanding Q3/Q4 confirmation; "
            f"{args.source_aggregation} aggregation of ECMWF+DWD+GEM+KMA "
            "causal operational forecasts"
        ),
        "contract": {
            "selection_fold": "train through 2024-Q1 -> select on Q2",
            "confirmation_folds": [
                "train through Q2 -> Q3",
                "train through Q3 -> Q4",
            ],
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "maximum_per_row_movement_ratio": 0.05,
            "source_aggregation": args.source_aggregation,
        },
        "sources": {
            "context_2024": [
                path.relative_to(ROOT).as_posix() for path in context_2024
            ],
            "context_2025": [
                path.relative_to(ROOT).as_posix() for path in context_2025
            ],
            "active_candidate": args.active_candidate,
        },
        "manifest_validation": manifest_validation,
        "coverage": {
            "validation_start": common_2024.min().isoformat(),
            "validation_end": common_2024.max().isoformat(),
            "validation_rows": int(len(common_2024)),
            "production_rows": int(len(features_2025)),
            "feature_count": int(features_2024.shape[1]),
            "independent_source_feature_counts": [
                int(frame.shape[1]) for frame in source_features_2024
            ],
        },
        "selected_specs": selected_specs,
        "validation": validation,
        "promoted_targets": promoted_targets,
        "expected_macro_score_delta_if_h2_transfers": float(
            expected_macro_delta
        ),
        "projected_public_score_if_h2_transfers": float(
            args.active_public_score + expected_macro_delta
        ),
        "candidate": candidate_record,
        "validation_cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(cache_path),
            "arrays": sorted(oof_cache),
        },
    }
    report_path = _rooted(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-2024", action="append", required=True)
    parser.add_argument("--manifest-2024", action="append", required=True)
    parser.add_argument("--context-2025", action="append", required=True)
    parser.add_argument("--manifest-2025", action="append", required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
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
        "--active-candidate",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument("--active-public-score", type=float, default=0.6461250914)
    parser.add_argument(
        "--source-aggregation",
        choices=("joint", "median", "mean"),
        default="joint",
        help=(
            "joint fits one concatenated model; median/mean fit one model per "
            "NWP source before robust aggregation"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output-submission",
        default=(
            "artifacts_final/candidates/"
            "multimodel_expanding_strict_20260727.csv"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "multimodel_expanding_20260727.json"
        ),
    )
    parser.add_argument(
        "--output-cache",
        default=(
            "artifacts_final/lineage/"
            "multimodel_expanding_20260727.npz"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected_specs": report["selected_specs"],
                "promoted_targets": report["promoted_targets"],
                "expected_macro_score_delta": report[
                    "expected_macro_score_delta_if_h2_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_h2_transfers"
                ],
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
