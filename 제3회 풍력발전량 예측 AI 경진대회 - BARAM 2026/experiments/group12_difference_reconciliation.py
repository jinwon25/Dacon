"""Zero-sum reconciliation of the group-1/group-2 differential component.

The active forecasts for groups 1 and 2 are more correlated than the observed
power series.  This experiment forecasts their capacity-normalized difference
from the supplied LDAPS/GFS site-IDW features, then makes equal and opposite
adjustments.  The active two-group total is therefore preserved exactly.

Validation follows a one-year-forward contract:

* train the differential expert on 2023;
* choose only the small reconciliation weight on 2024 Q1;
* inspect Q2/H2, months, seeds, and complete NWP issue-cycle bootstrap later;
* train on 2023+2024 for 2025 only if the validation promotion gate passes.
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
from experiments.blocked_rolling_validation import assign_issue_blocks
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("kpx_group_1", "kpx_group_2")
CAPACITY = CAPACITY_KWH["kpx_group_1"]
SEEDS = (13, 41, 79)
TIME_COLUMNS = (
    "hour",
    "month",
    "dayofweek",
    "lead_hour",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)
DEFAULT_WEIGHTS = (
    0.0125,
    0.025,
    0.0375,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pairwise_features(features: pd.DataFrame) -> pd.DataFrame:
    """Build symmetric common/difference features for the two equal-size groups."""
    output: dict[str, pd.Series] = {}
    marker_1 = "__kpx_group_1__"
    marker_2 = "__kpx_group_2__"
    for column in features.columns:
        if marker_1 not in column or not column.endswith("__idw"):
            continue
        counterpart = column.replace(marker_1, marker_2)
        if counterpart not in features.columns:
            continue
        stem = column.replace(marker_1, "__pair__")
        group_1 = features[column].astype(float)
        group_2 = features[counterpart].astype(float)
        output[f"mean__{stem}"] = 0.5 * (group_1 + group_2)
        output[f"delta__{stem}"] = group_1 - group_2
    for column in TIME_COLUMNS:
        if column in features.columns:
            output[column] = features[column].astype(float)
    if len(output) < 30:
        raise ValueError("insufficient paired group-1/group-2 IDW features")
    result = pd.DataFrame(output, index=features.index).astype("float32")
    if result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("pairwise reconciliation features are incomplete")
    return result


def make_difference_model(seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    """Fixed robust point model; no H2 or public-score hyperparameter search."""
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=-1,
        min_child_samples=96,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_difference_ensemble(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = (
        train_labels[GROUPS[0]].reindex(train_features.index).to_numpy(dtype=float)
        / CAPACITY
        - train_labels[GROUPS[1]]
        .reindex(train_features.index)
        .to_numpy(dtype=float)
        / CAPACITY
    )
    observed = np.isfinite(target)
    if int(observed.sum()) < 8_000:
        raise ValueError("difference model has insufficient observed labels")
    predictions: list[np.ndarray] = []
    for seed in seeds:
        model = make_difference_model(seed, n_estimators)
        model.fit(
            train_features.loc[observed],
            target[observed],
            callbacks=[lgb.log_evaluation(0)],
        )
        predictions.append(
            np.clip(
                np.asarray(model.predict(query_features), dtype=float),
                -1.0,
                1.0,
            )
        )
    members = np.vstack(predictions)
    return np.median(members, axis=0), members


def reconcile_pair(
    group_1: np.ndarray,
    group_2: np.ndarray,
    expert_normalized_difference: np.ndarray,
    *,
    weight: float,
    capacity: float = CAPACITY,
    movement_cap_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a bounded equal-and-opposite correction while preserving the sum."""
    first = np.asarray(group_1, dtype=float)
    second = np.asarray(group_2, dtype=float)
    expert = np.asarray(expert_normalized_difference, dtype=float)
    if first.shape != second.shape or first.shape != expert.shape:
        raise ValueError("reconciliation inputs have different shapes")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be within [0, 1]")
    if not 0.0 < movement_cap_ratio <= 0.10:
        raise ValueError("movement cap ratio must be within (0, 0.10]")

    base_difference = (first - second) / capacity
    raw_adjustment = (
        0.5 * weight * (expert - base_difference) * capacity
    )
    raw_adjustment = np.clip(
        raw_adjustment,
        -movement_cap_ratio * capacity,
        movement_cap_ratio * capacity,
    )
    lower = np.maximum(-first, second - capacity)
    upper = np.minimum(capacity - first, second)
    adjustment = np.clip(raw_adjustment, lower, upper)
    reconciled_first = first + adjustment
    reconciled_second = second - adjustment
    if not np.allclose(
        reconciled_first + reconciled_second,
        first + second,
        atol=1e-8,
        rtol=0.0,
    ):
        raise AssertionError("zero-sum reconciliation failed")
    return reconciled_first, reconciled_second, adjustment


def evaluate_pair(
    truth: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, float]:
    rows = np.asarray(rows, dtype=bool)
    metrics = {
        target: evaluate_group(
            truth[target][rows],
            prediction[target][rows],
            CAPACITY_KWH[target],
        )
        for target in GROUPS
    }
    return {
        "score": float(np.mean([row.score for row in metrics.values()])),
        "one_minus_nmae": float(
            np.mean([row.one_minus_nmae for row in metrics.values()])
        ),
        "ficr": float(np.mean([row.ficr for row in metrics.values()])),
    }


def pair_delta(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, float]:
    base = evaluate_pair(truth, reference, rows)
    treatment = evaluate_pair(truth, candidate, rows)
    return {
        component: float(treatment[component] - base[component])
        for component in ("score", "one_minus_nmae", "ficr")
    }


def select_weight(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record["q1_delta"]["score"] > 0.0
        and record["q1_delta"]["one_minus_nmae"] >= 0.0
        and record["q1_delta"]["ficr"] >= 0.0
    ]
    if not eligible:
        return {
            "weight": 0.0,
            "selection_eligible": False,
            "reason": "no positive Q1 weight improved both metric components",
        }
    selected = max(
        eligible,
        key=lambda record: (
            record["q1_delta"]["score"],
            min(record["q1_delta"].values()),
            -record["weight"],
        ),
    )
    return {
        "weight": float(selected["weight"]),
        "selection_eligible": True,
        "reason": "best Q1 score among weights improving both components",
    }


def pair_issue_bootstrap(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    issues: pd.DatetimeIndex,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    _, seasons = assign_issue_blocks(index, issues)
    issue_values = np.asarray(issues)
    finite_issue = ~pd.isna(issues)
    positions: dict[tuple[str, np.datetime64], np.ndarray] = {}
    stratum_issues: dict[str, np.ndarray] = {}
    for season in sorted(set(seasons[finite_issue])):
        available = np.unique(
            issue_values[finite_issue & (seasons == season)]
        )
        stratum_issues[season] = available
        for issue in available:
            positions[(season, issue)] = np.flatnonzero(
                finite_issue
                & (seasons == season)
                & (issue_values == issue)
            )
    if not stratum_issues:
        raise ValueError("no finite issue cycles are available for bootstrap")
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_rows: list[np.ndarray] = []
        for season, available in stratum_issues.items():
            sampled = rng.choice(
                available,
                size=len(available),
                replace=True,
            )
            sampled_rows.extend(positions[(season, issue)] for issue in sampled)
        rows = np.concatenate(sampled_rows)
        mask = np.zeros(len(index), dtype=bool)
        # Duplicate issue cycles must retain multiplicity, so evaluate directly
        # on the sampled integer positions rather than converting to a mask.
        base_metrics = {
            target: evaluate_group(
                truth[target][rows],
                reference[target][rows],
                CAPACITY_KWH[target],
            )
            for target in GROUPS
        }
        candidate_metrics = {
            target: evaluate_group(
                truth[target][rows],
                candidate[target][rows],
                CAPACITY_KWH[target],
            )
            for target in GROUPS
        }
        del mask
        values[iteration] = np.mean(
            [
                candidate_metrics[target].score
                - base_metrics[target].score
                for target in GROUPS
            ]
        )
    return {
        "n_bootstrap": int(n_bootstrap),
        "evaluated_rows": int(finite_issue.sum()),
        "excluded_missing_issue_rows": int((~finite_issue).sum()),
        "positive_fraction": float(np.mean(values > 0.0)),
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-15:
        return None
    return float(np.dot(first, second) / denominator)


def run(args: argparse.Namespace) -> dict[str, Any]:
    feature_path = _rooted(args.feature_cache)
    test_feature_path = _rooted(args.test_feature_cache)
    label_path = _rooted(args.labels)
    primary_path = _rooted(args.primary_cache)
    residual_path = _rooted(args.residual_cache)
    group3_path = _rooted(args.group3_cache)
    incumbent_path = _rooted(args.incumbent)
    weights = tuple(
        float(value.strip())
        for value in args.weights.split(",")
        if value.strip()
    )
    if (
        not weights
        or tuple(sorted(set(weights))) != weights
        or any(weight <= 0.0 or weight > 0.20 for weight in weights)
    ):
        raise ValueError("weights must be increasing and within (0, 0.20]")

    baselines, truths_series, index, issue_series = (
        load_frozen_validation_baselines(
            primary_path,
            residual_path,
            group3_path,
        )
    )
    issues = pd.DatetimeIndex(issue_series)
    raw_features = pd.read_pickle(feature_path)
    pair_features = build_pairwise_features(raw_features)
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").sort_index()

    validation_features = pair_features.reindex(index)
    if validation_features.isna().any().any():
        raise ValueError("validation feature cache does not cover OOF index")
    train_mask = (
        (pair_features.index >= pd.Timestamp("2023-01-01"))
        & (pair_features.index < pd.Timestamp("2024-01-01"))
    )
    expert, members = fit_difference_ensemble(
        pair_features.loc[train_mask],
        labels,
        validation_features,
        seeds=SEEDS,
        n_estimators=args.n_estimators,
    )
    truth = {
        target: truths_series[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    reference = {
        target: baselines[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    base_difference = (
        reference[GROUPS[0]] - reference[GROUPS[1]]
    ) / CAPACITY
    if args.uncertainty_gate == "unanimous":
        member_actions = members - base_difference
        gate = (np.min(member_actions, axis=0) > 0.0) | (
            np.max(member_actions, axis=0) < 0.0
        )
        expert = np.where(gate, expert, base_difference)
    else:
        gate = np.ones(len(index), dtype=bool)
    periods = {
        "q1": np.asarray(index < pd.Timestamp("2024-04-01")),
        "q2": np.asarray(
            (index >= pd.Timestamp("2024-04-01"))
            & (index < pd.Timestamp("2024-07-01"))
        ),
        "h2": np.asarray(index >= pd.Timestamp("2024-07-01")),
        "full": np.ones(len(index), dtype=bool),
    }

    weight_records: list[dict[str, Any]] = []
    candidates: dict[float, dict[str, np.ndarray]] = {}
    adjustments: dict[float, np.ndarray] = {}
    for weight in weights:
        group_1, group_2, adjustment = reconcile_pair(
            reference[GROUPS[0]],
            reference[GROUPS[1]],
            expert,
            weight=weight,
            movement_cap_ratio=args.movement_cap_ratio,
        )
        candidate = {GROUPS[0]: group_1, GROUPS[1]: group_2}
        candidates[weight] = candidate
        adjustments[weight] = adjustment
        weight_records.append(
            {
                "weight": float(weight),
                "q1_delta": pair_delta(
                    truth,
                    reference,
                    candidate,
                    periods["q1"],
                ),
                "q1_monthly_deltas": {
                    str(month): pair_delta(
                        truth,
                        reference,
                        candidate,
                        np.asarray(
                            (index < pd.Timestamp("2024-04-01"))
                            & (index.month == month)
                        ),
                    )
                    for month in (1, 2, 3)
                },
            }
        )
    selection = select_weight(weight_records)
    selected_weight = float(selection["weight"])
    if selected_weight == 0.0:
        candidate = reference
        adjustment = np.zeros(len(index), dtype=float)
    else:
        candidate = candidates[selected_weight]
        adjustment = adjustments[selected_weight]

    period_deltas = {
        name: pair_delta(truth, reference, candidate, rows)
        for name, rows in periods.items()
    }
    monthly_deltas = {
        str(month): pair_delta(
            truth,
            reference,
            candidate,
            np.asarray(index.month == month),
        )
        for month in range(1, 13)
    }
    seed_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for seed, member in zip(SEEDS, members):
        first, second, _ = reconcile_pair(
            reference[GROUPS[0]],
            reference[GROUPS[1]],
            member,
            weight=selected_weight,
            movement_cap_ratio=args.movement_cap_ratio,
        )
        seed_candidate = {GROUPS[0]: first, GROUPS[1]: second}
        seed_deltas[str(seed)] = {
            name: pair_delta(truth, reference, seed_candidate, rows)
            for name, rows in periods.items()
        }
    bootstrap = pair_issue_bootstrap(
        truth,
        reference,
        candidate,
        index,
        issues,
        n_bootstrap=args.n_bootstrap,
        seed=20260728,
    )

    primary_cache = np.load(primary_path, allow_pickle=False)
    residual_factor = (
        reference[GROUPS[0]]
        - primary_cache[f"{GROUPS[0]}__candidate"].astype(float)
    )
    pooled_factor = (
        reference[GROUPS[1]]
        - primary_cache[f"{GROUPS[1]}__reference"].astype(float)
    )
    reconciliation_vector = np.concatenate((adjustment, -adjustment))
    factor_vector = np.concatenate((residual_factor, pooled_factor))
    positive_months = int(
        sum(row["score"] > 0.0 for row in monthly_deltas.values())
    )
    all_seed_h2_scores_positive = all(
        row["h2"]["score"] > 0.0 for row in seed_deltas.values()
    )
    all_seed_period_scores_positive = all(
        period["score"] > 0.0
        for row in seed_deltas.values()
        for period in row.values()
    )
    common_confirmation_pass = bool(
        selection["selection_eligible"]
        and all(value >= 0.0 for value in period_deltas["full"].values())
        and all(value >= 0.0 for value in period_deltas["q2"].values())
        and all(value >= 0.0 for value in period_deltas["h2"].values())
        and all_seed_period_scores_positive
        and bootstrap["q05"] >= 0.0
        and bootstrap["positive_fraction"] >= 0.90
    )
    strict_pass = bool(
        common_confirmation_pass
        and positive_months >= 9
    )
    controlled_pass = bool(
        common_confirmation_pass
        and positive_months >= 7
        and selected_weight not in (min(weights), max(weights))
        and abs(
            _cosine(reconciliation_vector, factor_vector) or 0.0
        )
        <= 0.30
    )

    candidate_record: dict[str, Any] | None = None
    if args.write_candidate:
        if not controlled_pass:
            raise RuntimeError(
                "candidate generation blocked because controlled promotion failed"
            )
        test_raw = pd.read_pickle(test_feature_path)
        test_features = build_pairwise_features(test_raw)
        production_train = (
            (pair_features.index >= pd.Timestamp("2023-01-01"))
            & (pair_features.index < pd.Timestamp("2025-01-01"))
        )
        production_expert, production_members = fit_difference_ensemble(
            pair_features.loc[production_train],
            labels,
            test_features,
            seeds=SEEDS,
            n_estimators=args.n_estimators,
        )
        output = pd.read_csv(incumbent_path, encoding="utf-8-sig")
        output_index = pd.DatetimeIndex(
            pd.to_datetime(output["forecast_kst_dtm"])
        )
        aligned_expert = pd.Series(
            production_expert,
            index=test_features.index,
        ).reindex(output_index)
        if aligned_expert.isna().any():
            raise ValueError("production expert does not align to submission")
        aligned_members = np.vstack(
            [
                pd.Series(member, index=test_features.index)
                .reindex(output_index)
                .to_numpy(dtype=float)
                for member in production_members
            ]
        )
        if not np.isfinite(aligned_members).all():
            raise ValueError("production ensemble does not align to submission")
        if args.uncertainty_gate == "unanimous":
            output_base_difference = (
                output[GROUPS[0]].to_numpy(dtype=float)
                - output[GROUPS[1]].to_numpy(dtype=float)
            ) / CAPACITY
            member_actions = aligned_members - output_base_difference
            production_gate = (np.min(member_actions, axis=0) > 0.0) | (
                np.max(member_actions, axis=0) < 0.0
            )
            aligned_expert = pd.Series(
                np.where(
                    production_gate,
                    aligned_expert.to_numpy(dtype=float),
                    output_base_difference,
                ),
                index=output_index,
            )
        first, second, _ = reconcile_pair(
            output[GROUPS[0]].to_numpy(dtype=float),
            output[GROUPS[1]].to_numpy(dtype=float),
            aligned_expert.to_numpy(dtype=float),
            weight=selected_weight,
            movement_cap_ratio=args.movement_cap_ratio,
        )
        original_first = output[GROUPS[0]].to_numpy(dtype=float).copy()
        original_second = output[GROUPS[1]].to_numpy(dtype=float).copy()
        original_group3 = output["kpx_group_3"].to_numpy(dtype=float).copy()
        output[GROUPS[0]] = first
        output[GROUPS[1]] = second
        output_path = _rooted(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(f"candidate validation failed: {audit.errors}")
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
            "promotion_tier": (
                "strict" if strict_pass else "controlled_exploratory"
            ),
            "composition": {
                "group3_identical_to_incumbent": bool(
                    np.array_equal(
                        output["kpx_group_3"].to_numpy(dtype=float),
                        original_group3,
                    )
                ),
                "maximum_group12_sum_difference_kwh": float(
                    np.max(
                        np.abs(
                            (
                                output[GROUPS[0]].to_numpy(dtype=float)
                                + output[GROUPS[1]].to_numpy(dtype=float)
                            )
                            - (original_first + original_second)
                        )
                    )
                ),
                "changed_rows": int(
                    np.sum(
                        np.abs(
                            output[GROUPS[0]].to_numpy(dtype=float)
                            - original_first
                        )
                        > 1e-8
                    )
                ),
            },
        }

    report = {
        "family": "group12_zero_sum_difference_reconciliation",
        "method": (
            "robust LightGBM ensemble predicts normalized G1-G2 power "
            "difference from supplied LDAPS/GFS paired site-IDW features"
        ),
        "contract": {
            "validation_training": "2023 only",
            "validation_query": "2024 exact active OOF surface",
            "weight_selection": "2024 Q1 only",
            "confirmation": (
                "Q2, H2, months, three seeds, complete issue-cycle bootstrap"
            ),
            "production_training": "2023+2024 only",
            "active_group12_sum_preserved": True,
            "group3_unchanged": True,
            "public_score_used_for_weight_or_gate_selection": False,
            "repeated_2024_benchmark_used_for_method_development": True,
            "movement_cap_ratio_per_group": args.movement_cap_ratio,
            "uncertainty_gate": args.uncertainty_gate,
        },
        "data_diagnostic": {
            "observed_group_correlation": float(
                np.corrcoef(
                    truth[GROUPS[0]] / CAPACITY,
                    truth[GROUPS[1]] / CAPACITY,
                )[0, 1]
            ),
            "active_forecast_group_correlation": float(
                np.corrcoef(
                    reference[GROUPS[0]] / CAPACITY,
                    reference[GROUPS[1]] / CAPACITY,
                )[0, 1]
            ),
            "active_normalized_difference_mae": float(
                np.mean(
                    np.abs(
                        (
                            truth[GROUPS[0]] - truth[GROUPS[1]]
                        )
                        / CAPACITY
                        - (
                            reference[GROUPS[0]]
                            - reference[GROUPS[1]]
                        )
                        / CAPACITY
                    )
                )
            ),
            "pairwise_feature_count": int(pair_features.shape[1]),
            "validation_gate_coverage": float(np.mean(gate)),
        },
        "selection": {
            **selection,
            "records": weight_records,
        },
        "validation": {
            "period_deltas": period_deltas,
            "monthly_deltas": monthly_deltas,
            "positive_score_months": positive_months,
            "seed_period_deltas": seed_deltas,
            "all_seed_h2_scores_positive": all_seed_h2_scores_positive,
            "all_seed_period_scores_positive": (
                all_seed_period_scores_positive
            ),
            "issue_block_bootstrap": bootstrap,
        },
        "orthogonality": {
            "cosine_vs_retained_public_factor_vector": _cosine(
                reconciliation_vector,
                factor_vector,
            ),
            "mean_absolute_movement_capacity_ratio": float(
                np.mean(np.abs(adjustment)) / CAPACITY
            ),
            "maximum_absolute_movement_capacity_ratio": float(
                np.max(np.abs(adjustment)) / CAPACITY
            ),
            "interpretation": (
                "near-zero cosine indicates a new differential axis rather "
                "than simple scaling of retained G1/G2 public factors"
            ),
        },
        "promotion": {
            "strict_passed": strict_pass,
            "controlled_exploratory_passed": controlled_pass,
            "candidate_written": candidate_record is not None,
            "decision": (
                "eligible_strict"
                if strict_pass
                else (
                    "eligible_controlled_exploratory"
                    if controlled_pass
                    else "hold_no_candidate"
                )
            ),
            "strict_failure": (
                None
                if strict_pass
                else (
                    f"positive score months {positive_months}/12 < 9/12"
                )
            ),
        },
        "candidate": candidate_record,
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--test-feature-cache",
        default="artifacts_final/feature_cache/features_test.pkl",
    )
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
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--weights",
        default=",".join(str(value) for value in DEFAULT_WEIGHTS),
    )
    parser.add_argument("--movement-cap-ratio", type=float, default=0.02)
    parser.add_argument(
        "--uncertainty-gate",
        choices=("none", "unanimous"),
        default="none",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--write-candidate", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "group12_difference_reconciliation_unanimous_w10_20260728.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "group12_difference_reconciliation_20260728.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selection": report["selection"]["weight"],
                "period_deltas": report["validation"]["period_deltas"],
                "bootstrap": report["validation"]["issue_block_bootstrap"],
                "orthogonality": report["orthogonality"],
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
