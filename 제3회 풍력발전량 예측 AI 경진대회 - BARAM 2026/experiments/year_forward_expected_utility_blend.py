"""One-year-forward Bayes actions for the exact DACON settlement utility.

This experiment estimates a conditional generation distribution with quantile
LightGBM models trained on the prior calendar year.  It converts that
distribution to a point action by maximizing the expected official objective,
including the 10% eligibility rule and the 6%/8% FiCR cliffs.

The action is only a bounded, sparse increment over the current candidate.
Coverage and shrinkage are selected on 2024 H1 with separate Q1/Q2 component
guards; H2, model seeds, months, and complete forecast-issue blocks are then
confirmation evidence.  No public score is used for policy selection.
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

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.kma_base_v2_local_overlay import OverlayPolicy, apply_overlay
from experiments.kma_year_forward_quantile_blend import (
    H2_START,
    Q2_START,
    SEEDS,
    VALIDATION_END,
    load_context_feature_bundle,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("kpx_group_1", "kpx_group_2")
SUPPORTED_TARGETS = tuple(CAPACITY_KWH)
QUANTILE_LEVELS = np.asarray(
    [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    dtype=float,
)
TOP_FRACTIONS = (0.05, 0.075, 0.10, 0.15)
BLEND_WEIGHTS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
INCREMENTAL_MOVEMENT_CAP = 0.025
TOTAL_MOVEMENT_CAPS = {
    "kpx_group_1": 0.05,
    "kpx_group_2": 0.06,
    "kpx_group_3": 0.05,
}
CALENDAR_COLUMNS = {
    "hour",
    "month",
    "dayofweek",
    "lead_hour",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
}
PHYSICAL_TOKENS = (
    "ws",
    "10u",
    "10v",
    "MU",
    "MV",
    "gust",
    "_u__",
    "_v__",
    "surface_0_sp",
    "prmsl",
    "blh",
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    if not targets:
        raise ValueError("at least one target is required")
    unknown = set(targets).difference(SUPPORTED_TARGETS)
    if unknown:
        raise ValueError(f"unsupported targets: {sorted(unknown)}")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    return targets


def select_distribution_columns(
    features: pd.DataFrame, target: str
) -> list[str]:
    """Compact physical aggregates plus the target's IDW features."""
    columns = [
        column
        for column in features.columns
        if column in CALENDAR_COLUMNS
        or f"__{target}__" in column
        or (
            any(
                suffix in column
                for suffix in ("__mean", "__std", "__min", "__max")
            )
            and any(token in column for token in PHYSICAL_TOKENS)
        )
    ]
    if not columns:
        raise ValueError(f"no distribution features selected for {target}")
    return columns


def build_distribution_features(
    base_features: pd.DataFrame,
    context: pd.DataFrame,
    target: str,
    year: int,
) -> pd.DataFrame:
    rows = (base_features.index >= pd.Timestamp(f"{year}-01-01")) & (
        base_features.index < pd.Timestamp(f"{year + 1}-01-01")
    )
    base = base_features.loc[
        rows, select_distribution_columns(base_features, target)
    ]
    external = context.reindex(base.index).add_prefix("external__")
    return pd.concat([base, external], axis=1).astype("float32")


def make_quantile_model(
    alpha: float, seed: int, n_estimators: int
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.75,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_distribution(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    query_features: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
    capacity: float,
) -> np.ndarray:
    observed = train_target.notna()
    if int(observed.sum()) < 5_000:
        raise ValueError("distribution model has insufficient training labels")
    predictions: list[np.ndarray] = []
    for quantile_i, alpha in enumerate(QUANTILE_LEVELS):
        model = make_quantile_model(
            alpha, seed + 100 * quantile_i, n_estimators
        )
        model.fit(
            train_features.loc[observed],
            train_target.loc[observed],
            callbacks=[lgb.log_evaluation(0)],
        )
        predictions.append(
            np.clip(model.predict(query_features), 0.0, capacity)
        )
    return np.sort(np.column_stack(predictions), axis=1)


def expected_official_utility(
    actions: np.ndarray,
    samples: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
) -> np.ndarray:
    """Expected row utility proportional to the official group objective."""
    actions = np.asarray(actions, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if actions.ndim != 2 or samples.ndim != 2:
        raise ValueError("actions and samples must both be two-dimensional")
    if actions.shape[0] != samples.shape[0]:
        raise ValueError("actions and samples must have the same row count")
    if mean_eligible_generation <= 0.0:
        raise ValueError("mean eligible generation must be positive")
    error = np.abs(actions[:, :, None] - samples[:, None, :])
    eligible = samples[:, None, :] >= 0.10 * capacity
    units = np.where(
        error <= 0.06 * capacity,
        4.0,
        np.where(error <= 0.08 * capacity, 3.0, 0.0),
    )
    contribution = np.where(
        eligible,
        -0.5 * error / capacity
        + 0.5
        * samples[:, None, :]
        * units
        / (4.0 * mean_eligible_generation),
        0.0,
    )
    return np.mean(contribution, axis=2)


def choose_bayes_action(
    samples: np.ndarray,
    reference: np.ndarray,
    *,
    capacity: float,
    mean_eligible_generation: float,
    batch_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize sample-average official utility over all cliff breakpoints."""
    samples = np.sort(np.asarray(samples, dtype=float), axis=1)
    reference = np.asarray(reference, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != len(reference):
        raise ValueError("samples and reference do not align")
    output = np.empty(len(reference), dtype=float)
    advantage = np.empty(len(reference), dtype=float)
    offsets = np.asarray([-0.08, -0.06, 0.0, 0.06, 0.08]) * capacity
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        sample = samples[start:stop]
        incumbent = reference[start:stop]
        actions = np.concatenate(
            [
                incumbent[:, None],
                (sample[:, :, None] + offsets[None, None, :]).reshape(
                    len(sample), -1
                ),
            ],
            axis=1,
        )
        actions = np.clip(actions, 0.0, capacity)
        utility = expected_official_utility(
            actions,
            sample,
            capacity=capacity,
            mean_eligible_generation=mean_eligible_generation,
        )
        selected = np.argmax(utility, axis=1)
        output[start:stop] = actions[np.arange(len(sample)), selected]
        advantage[start:stop] = (
            utility[np.arange(len(sample)), selected] - utility[:, 0]
        )
    return output, advantage


def apply_rank_policy(
    incumbent: np.ndarray,
    underlying_reference: np.ndarray,
    action: np.ndarray,
    advantage: np.ndarray,
    *,
    top_fraction: float,
    weight: float,
    capacity: float,
    total_movement_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed annual advantage rank with bounded incremental movement."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top fraction must lie in (0, 1]")
    incumbent = np.asarray(incumbent, dtype=float)
    underlying_reference = np.asarray(underlying_reference, dtype=float)
    action = np.asarray(action, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    if not (
        incumbent.shape
        == underlying_reference.shape
        == action.shape
        == advantage.shape
    ):
        raise ValueError("rank-policy vectors must align")
    threshold = (
        -np.inf
        if top_fraction == 1.0
        else float(np.quantile(advantage, 1.0 - top_fraction))
    )
    gate = advantage >= threshold
    movement = np.clip(
        float(weight) * (action - incumbent),
        -INCREMENTAL_MOVEMENT_CAP * capacity,
        INCREMENTAL_MOVEMENT_CAP * capacity,
    )
    candidate = incumbent.copy()
    candidate[gate] += movement[gate]
    lower = np.clip(
        underlying_reference - total_movement_cap * capacity, 0.0, capacity
    )
    upper = np.clip(
        underlying_reference + total_movement_cap * capacity, 0.0, capacity
    )
    return np.clip(candidate, lower, upper), gate


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "q1": np.asarray(index < Q2_START),
        "q2": np.asarray((index >= Q2_START) & (index < H2_START)),
        "h1": np.asarray(index < H2_START),
        "h2": np.asarray(
            (index >= H2_START) & (index < VALIDATION_END)
        ),
        "full": np.asarray(index < VALIDATION_END),
    }


def _align_samples(
    samples: np.ndarray,
    sample_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
    fallback: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame(samples, index=sample_index).reindex(target_index)
    missing = frame.isna().any(axis=1)
    if int(missing.sum()) > 1:
        raise ValueError("distribution samples have more than one alignment gap")
    frame.loc[missing, :] = np.asarray(fallback)[missing.to_numpy(), None]
    return frame.to_numpy(dtype=float)


def _load_group2_incumbent(
    driver: np.lib.npyio.NpzFile,
    completed_member: np.lib.npyio.NpzFile,
    policy_report: Path,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    target = "kpx_group_2"
    index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{target}__valid_index_ns"])
    )
    member_index = pd.DatetimeIndex(
        pd.to_datetime(completed_member["index_ns"])
    )
    if not index.equals(member_index):
        raise ValueError("completed group-2 member and driver indexes differ")
    raw = json.loads(policy_report.read_text(encoding="utf-8"))["search"][
        "exploratory_by_group"
    ][target]["policy"]
    policy = OverlayPolicy(**{**raw, "alpha": 0.2375})
    reference = driver[f"{target}__exact_base"].astype(float)
    incumbent, _ = apply_overlay(
        reference,
        completed_member[f"{target}__candidate"].astype(float),
        policy,
    )
    return (
        index,
        driver[f"{target}__valid_truth"].astype(float),
        reference,
        incumbent,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    validation_only = bool(getattr(args, "validation_only", False))
    extra_contexts = {
        year: tuple(
            _rooted(value)
            for value in getattr(args, f"extra_context_{year}", ())
        )
        for year in ("2023", "2024", "2025")
    }
    base_features = pd.read_pickle(_rooted(args.train_features))
    test_features = pd.read_pickle(_rooted(args.test_features))
    context_2023, _ = load_context_feature_bundle(
        _rooted(args.context_2023),
        extra_contexts["2023"],
    )
    context_2024, issue_2024 = load_context_feature_bundle(
        _rooted(args.context_2024),
        extra_contexts["2024"],
    )
    context_2025 = None
    if not validation_only:
        context_2025, _ = load_context_feature_bundle(
            _rooted(args.context_2025),
            extra_contexts["2025"],
        )
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").reindex(base_features.index)
    driver = np.load(_rooted(args.driver), allow_pickle=False)
    completed_member = np.load(
        _rooted(args.completed_group2_member), allow_pickle=False
    )
    previous_cache = np.load(
        _rooted(args.previous_validation_cache), allow_pickle=False
    )
    kma_oof = np.load(_rooted(args.kma_oof), allow_pickle=False)

    validation: dict[str, Any] = {}
    production_specs: dict[str, dict[str, float]] = {}
    validation_cache: dict[str, np.ndarray] = {}
    expected_macro_increment = 0.0
    for target in args.targets:
        capacity = CAPACITY_KWH[target]
        train_2023 = build_distribution_features(
            base_features, context_2023, target, 2023
        )
        query_2024 = build_distribution_features(
            base_features, context_2024, target, 2024
        )
        common = train_2023.columns.intersection(query_2024.columns)
        train_2023 = train_2023[common]
        query_2024 = query_2024[common]
        target_2023 = labels[target].reindex(train_2023.index)
        seed_samples = [
            fit_distribution(
                train_2023,
                target_2023,
                query_2024,
                seed=seed,
                n_estimators=args.n_estimators,
                capacity=capacity,
            )
            for seed in SEEDS
        ]

        if target in {"kpx_group_1", "kpx_group_3"}:
            index = pd.DatetimeIndex(
                pd.to_datetime(previous_cache[f"{target}__index_ns"])
            )
            if target == "kpx_group_1":
                truth = driver[f"{target}__valid_truth"].astype(float)
                reference = driver[f"{target}__exact_base"].astype(float)
            else:
                kma_index = pd.DatetimeIndex(
                    pd.to_datetime(kma_oof["index_ns"])
                )
                if not kma_index.equals(index):
                    raise ValueError("group-3 KMA and validation indexes differ")
                truth = kma_oof["truth"].astype(float)
                reference = kma_oof["rolling_candidate"].astype(float)
            incumbent_key = (
                f"{target}__incumbent"
                if f"{target}__incumbent" in previous_cache.files
                else f"{target}__candidate"
            )
            incumbent = previous_cache[incumbent_key].astype(float)
        else:
            index, truth, reference, incumbent = _load_group2_incumbent(
                driver,
                completed_member,
                _rooted(args.group2_policy_report),
            )
        aligned_seed_samples = [
            _align_samples(
                item, query_2024.index, index, incumbent
            )
            for item in seed_samples
        ]
        ensemble_samples = np.concatenate(aligned_seed_samples, axis=1)
        observed = target_2023.notna()
        mean_generation = float(
            target_2023[
                observed & (target_2023 >= 0.10 * capacity)
            ].mean()
        )
        action, advantage = choose_bayes_action(
            ensemble_samples,
            incumbent,
            capacity=capacity,
            mean_eligible_generation=mean_generation,
        )
        periods = _periods(index)
        records: list[dict[str, Any]] = []
        candidates: dict[tuple[float, float], np.ndarray] = {}
        for top_fraction in TOP_FRACTIONS:
            for weight in BLEND_WEIGHTS:
                candidate, gate = apply_rank_policy(
                    incumbent,
                    reference,
                    action,
                    advantage,
                    top_fraction=top_fraction,
                    weight=weight,
                    capacity=capacity,
                    total_movement_cap=TOTAL_MOVEMENT_CAPS[target],
                )
                deltas = {
                    name: metric_delta(
                        truth,
                        incumbent,
                        candidate,
                        capacity,
                        rows,
                    )
                    for name, rows in periods.items()
                }
                record = {
                    "top_fraction": top_fraction,
                    "weight": weight,
                    "changed_rows": int(gate.sum()),
                    "period_deltas": deltas,
                    "h1_eligible": bool(
                        min(deltas["q1"].values()) > 0.0
                        and min(deltas["q2"].values()) > 0.0
                        and min(deltas["h1"].values()) > 0.0
                    ),
                }
                records.append(record)
                candidates[(top_fraction, weight)] = candidate
        eligible = [row for row in records if row["h1_eligible"]]
        selected = (
            max(
                eligible,
                key=lambda row: (
                    row["period_deltas"]["h1"]["score"],
                    min(row["period_deltas"]["q1"].values()),
                    min(row["period_deltas"]["q2"].values()),
                    -row["top_fraction"],
                    -row["weight"],
                ),
            )
            if eligible
            else None
        )
        if selected is None:
            validation[target] = {
                "feature_count": int(len(common)),
                "records": records,
                "selected": None,
                "promotion": "rejected_on_h1",
            }
            continue

        top_fraction = float(selected["top_fraction"])
        weight = float(selected["weight"])
        candidate = candidates[(top_fraction, weight)]
        seed_rows: list[dict[str, Any]] = []
        for seed, samples in zip(SEEDS, aligned_seed_samples):
            seed_action, seed_advantage = choose_bayes_action(
                samples,
                incumbent,
                capacity=capacity,
                mean_eligible_generation=mean_generation,
            )
            seed_candidate, _ = apply_rank_policy(
                incumbent,
                reference,
                seed_action,
                seed_advantage,
                top_fraction=top_fraction,
                weight=weight,
                capacity=capacity,
                total_movement_cap=TOTAL_MOVEMENT_CAPS[target],
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "period_deltas": {
                        name: metric_delta(
                            truth,
                            incumbent,
                            seed_candidate,
                            capacity,
                            rows,
                        )
                        for name, rows in periods.items()
                    },
                }
            )
        monthly = {
            str(month): metric_delta(
                truth,
                incumbent,
                candidate,
                capacity,
                periods["full"] & np.asarray(index.month == month),
            )
            for month in range(1, 13)
        }
        issue = issue_2024.reindex(index)
        if int(issue.loc[index < VALIDATION_END].isna().sum()):
            raise ValueError("KMA issue times are missing in the validation year")
        issue = issue.fillna(pd.Timestamp("2024-12-31 13:00")).to_numpy()
        bootstraps = {
            name: evaluate_blocked_rolling(
                truth,
                incumbent,
                candidate,
                index,
                issue,
                rows & (truth >= 0.10 * capacity),
                n_bootstrap=args.n_bootstrap,
                seed=20260726,
            )
            for name, rows in {
                "h2": periods["h2"],
                "full": periods["full"],
            }.items()
        }
        positive_months = int(
            sum(value["score"] > 0.0 for value in monthly.values())
        )
        gates = {
            "h1_q1_q2_components_positive": bool(selected["h1_eligible"]),
            "h2_components_positive": (
                min(selected["period_deltas"]["h2"].values()) > 0.0
            ),
            "every_seed_h2_components_positive": all(
                min(row["period_deltas"]["h2"].values()) > 0.0
                for row in seed_rows
            ),
            "every_seed_h1_score_positive": all(
                row["period_deltas"]["h1"]["score"] > 0.0
                for row in seed_rows
            ),
            "positive_month_fraction_at_least_75pct": (
                positive_months / 12.0 >= 0.75
            ),
            "h2_bootstrap_q05_positive": (
                bootstraps["h2"]["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "h2_bootstrap_positive_fraction_at_least_90pct": (
                bootstraps["h2"]["issue_block_bootstrap"][
                    "positive_fraction"
                ]
                >= 0.90
            ),
            "full_bootstrap_q05_positive": (
                bootstraps["full"]["issue_block_bootstrap"]["q05"] > 0.0
            ),
        }
        promoted = bool(all(gates.values()))
        if promoted:
            production_specs[target] = {
                "top_fraction": top_fraction,
                "weight": weight,
            }
            expected_macro_increment += (
                selected["period_deltas"]["full"]["score"] / 3.0
            )
        validation_cache[f"{target}__index_ns"] = (
            index.astype("int64").to_numpy()
        )
        validation_cache[f"{target}__action"] = action.astype("float32")
        validation_cache[f"{target}__advantage"] = advantage.astype("float32")
        validation_cache[f"{target}__candidate"] = candidate.astype("float32")
        validation[target] = {
            "feature_count": int(len(common)),
            "quantile_levels": QUANTILE_LEVELS.tolist(),
            "mean_eligible_generation": mean_generation,
            "records": records,
            "selected": selected,
            "seed_stability": seed_rows,
            "monthly_deltas": monthly,
            "positive_months": positive_months,
            "issue_block_validation": bootstraps,
            "gates": gates,
            "promotion": "promoted" if promoted else "rejected",
        }

    output_cache = _rooted(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, **validation_cache)

    candidate_record: dict[str, Any] | None = None
    if production_specs and not validation_only:
        if context_2025 is None:
            raise RuntimeError("production context was not loaded")
        current = pd.read_csv(
            _rooted(args.current_candidate), encoding="utf-8-sig"
        )
        underlying = pd.read_csv(
            _rooted(args.underlying_candidate), encoding="utf-8-sig"
        )
        if not current[["forecast_id", "forecast_kst_dtm"]].equals(
            underlying[["forecast_id", "forecast_kst_dtm"]]
        ):
            raise ValueError("current and underlying candidate IDs differ")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(current["forecast_kst_dtm"])
        )
        if not test_index.equals(test_features.index):
            raise ValueError("test features and current candidate indexes differ")
        output = current.copy()
        for target, spec in production_specs.items():
            capacity = CAPACITY_KWH[target]
            train_2024 = build_distribution_features(
                base_features, context_2024, target, 2024
            )
            query_2025 = build_distribution_features(
                test_features, context_2025, target, 2025
            )
            common = train_2024.columns.intersection(query_2025.columns)
            target_2024 = labels[target].reindex(train_2024.index)
            samples = np.concatenate(
                [
                    fit_distribution(
                        train_2024[common],
                        target_2024,
                        query_2025[common],
                        seed=seed,
                        n_estimators=args.n_estimators,
                        capacity=capacity,
                    )
                    for seed in SEEDS
                ],
                axis=1,
            )
            incumbent = current[target].to_numpy(dtype=float)
            reference = (
                underlying[target].to_numpy(dtype=float)
                if target == "kpx_group_3"
                else driver[f"{target}__test_exact_base"].astype(float)
            )
            observed = target_2024.notna()
            mean_generation = float(
                target_2024[
                    observed & (target_2024 >= 0.10 * capacity)
                ].mean()
            )
            action, advantage = choose_bayes_action(
                samples,
                incumbent,
                capacity=capacity,
                mean_eligible_generation=mean_generation,
            )
            output[target], _ = apply_rank_policy(
                incumbent,
                reference,
                action,
                advantage,
                top_fraction=spec["top_fraction"],
                weight=spec["weight"],
                capacity=capacity,
                total_movement_cap=TOTAL_MOVEMENT_CAPS[target],
            )

        output_path = _rooted(args.output_submission)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")
        movement = {
            target: np.abs(
                output[target].to_numpy(dtype=float)
                - current[target].to_numpy(dtype=float)
            )
            for target in CAPACITY_KWH
        }
        normalized = np.concatenate(
            [
                movement[target] / CAPACITY_KWH[target]
                for target in CAPACITY_KWH
            ]
        )
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
            "movement": {
                "changed_target_cell_ratio": float(
                    np.mean(normalized > (1e-6 / 21_600.0))
                ),
                "p95_ratio": float(np.quantile(normalized, 0.95)),
                "maximum_ratio": float(normalized.max()),
                "by_group": {
                    target: {
                        "changed_rows": int(
                            np.sum(movement[target] > 1e-6)
                        ),
                        "mean_kwh": float(movement[target].mean()),
                        "p95_kwh": float(
                            np.quantile(movement[target], 0.95)
                        ),
                        "maximum_kwh": float(movement[target].max()),
                    }
                    for target in CAPACITY_KWH
                },
            },
        }

    report = {
        "family": "one_year_forward_expected_official_utility",
        "method": (
            "prior-year conditional quantiles -> sample-average Bayes action "
            "for official NMAE/FiCR utility -> sparse bounded incumbent blend"
        ),
        "contract": {
            "validation_only": validation_only,
            "training": "2023 -> 2024 validation; 2024 -> 2025 production",
            "selection": (
                "H1 only; Q1 and Q2 must separately improve score, "
                "1-NMAE, and FiCR"
            ),
            "confirmation": "H2, three seeds, months, issue-cycle bootstrap",
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "incremental_movement_cap": INCREMENTAL_MOVEMENT_CAP,
            "total_movement_caps": TOTAL_MOVEMENT_CAPS,
        },
        "validation": validation,
        "targets": list(args.targets),
        "production_specs": production_specs,
        "expected_macro_increment_if_2024_transfers": float(
            expected_macro_increment
        ),
        "projected_public_score_if_all_local_deltas_transfer": float(
            args.current_candidate_projection + expected_macro_increment
        ),
        "cache": {
            "path": output_cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_cache),
        },
        "candidate": candidate_record,
    }
    output_report = _rooted(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=DEFAULT_TARGETS,
        help="comma-separated groups to evaluate",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="validate 2023 -> 2024 without loading 2025 context or writing a candidate",
    )
    parser.add_argument(
        "--train-features",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--test-features",
        default="artifacts_final/feature_cache/features_test.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--kma-oof",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--completed-group2-member",
        default="artifacts_final/lineage/base_v2_group2_complete_oof.npz",
    )
    parser.add_argument(
        "--group2-policy-report",
        default=(
            "artifacts_final/base_v2/"
            "kma_incumbent_local_overlay_20260725.json"
        ),
    )
    parser.add_argument(
        "--previous-validation-cache",
        default="artifacts_final/lineage/kma_spatial_context_screen_20260726.npz",
    )
    for year in ("2023", "2024", "2025"):
        parser.add_argument(
            f"--context-{year}",
            default=(
                "artifacts_final/external_weather/"
                f"kma_um_regional_context_{year}/features.csv"
            ),
        )
        parser.add_argument(
            f"--extra-context-{year}",
            action="append",
            default=[],
            help="optional additional causal context CSV; may be repeated",
        )
    parser.add_argument(
        "--current-candidate",
        default=(
            "artifacts_final/candidates/"
            "kma_year_forward_g1w10_g3w075_20260726.csv"
        ),
    )
    parser.add_argument(
        "--underlying-candidate",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
        help="candidate before the year-forward group-3 overlay",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--current-candidate-projection", type=float, default=0.6466138600
    )
    parser.add_argument(
        "--output-cache",
        default=(
            "artifacts_final/lineage/"
            "year_forward_expected_utility_20260726.npz"
        ),
    )
    parser.add_argument(
        "--output-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_year_forward_expected_utility_20260726.csv"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "year_forward_expected_utility_20260726.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "production_specs": report["production_specs"],
                "expected_macro_increment": report[
                    "expected_macro_increment_if_2024_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_all_local_deltas_transfer"
                ],
                "validation": {
                    target: {
                        "selected": value.get("selected"),
                        "positive_months": value.get("positive_months"),
                        "gates": value.get("gates"),
                        "promotion": value["promotion"],
                    }
                    for target, value in report["validation"].items()
                },
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
