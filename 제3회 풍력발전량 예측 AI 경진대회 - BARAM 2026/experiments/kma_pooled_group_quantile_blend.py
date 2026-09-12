"""Pooled one-year-forward quantile model for nearby wind-farm groups.

Groups 1 and 3 share one capacity-normalized wind-power model.  Target
one-hot columns retain group-specific power curves while the pooled sample
reduces year-forward estimation variance.  The quantile is fixed before
validation; only a bounded blend weight is selected on 2024 Q1.  Q2, H2,
model seeds, months, and complete forecast-issue blocks remain confirmation
evidence.
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
from experiments.kma_base_v2_local_overlay import OverlayPolicy
from experiments.kma_year_forward_quantile_blend import (
    ALPHAS,
    PARSIMONY_SCORE_TOLERANCE,
    SEEDS,
    VALIDATION_END,
    _align_prediction,
    _periods,
    _validation_surfaces,
    apply_bounded_blend,
    evaluate_weights,
    load_context_feature_bundle,
    make_quantile_model,
    metric_delta,
    parse_targets,
    select_weight,
)
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("kpx_group_1", "kpx_group_3")
SUPPORTED_TARGETS = tuple(CAPACITY_KWH)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_pooled_design(
    features: pd.DataFrame,
    targets: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, slice]]:
    """Stack one feature block per target and append deterministic one-hot IDs."""
    blocks: list[pd.DataFrame] = []
    slices: dict[str, slice] = {}
    start = 0
    for target in targets:
        block = features.reset_index(drop=True).copy()
        for supported in targets:
            block[f"pooled_group__{supported}"] = float(target == supported)
        stop = start + len(block)
        slices[target] = slice(start, stop)
        blocks.append(block)
        start = stop
    pooled = pd.concat(blocks, axis=0, ignore_index=True)
    if pooled.isna().any().any() or not np.isfinite(pooled.to_numpy()).all():
        raise ValueError("pooled context features are incomplete")
    return pooled.astype("float32"), slices


def make_pooled_target(
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
    targets: tuple[str, ...],
) -> pd.Series:
    values = [
        labels[target].reindex(index).to_numpy(dtype=float)
        / CAPACITY_KWH[target]
        for target in targets
    ]
    return pd.Series(np.concatenate(values), dtype=float)


def fit_pooled_expert(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    query_features: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    alpha: float,
    seed: int,
    n_estimators: int,
) -> dict[str, np.ndarray]:
    pooled_train, _ = make_pooled_design(train_features, targets)
    pooled_query, query_slices = make_pooled_design(query_features, targets)
    pooled_target = make_pooled_target(
        train_labels,
        pd.DatetimeIndex(train_features.index),
        targets,
    )
    observed = pooled_target.notna()
    if int(observed.sum()) < 10_000:
        raise ValueError("pooled model has insufficient observed labels")
    model = make_quantile_model(alpha, seed, n_estimators)
    model.fit(
        pooled_train.loc[observed],
        pooled_target.loc[observed],
        callbacks=[lgb.log_evaluation(0)],
    )
    normalized = np.clip(
        np.asarray(model.predict(pooled_query), dtype=float),
        0.0,
        1.0,
    )
    return {
        target: normalized[query_slices[target]] * CAPACITY_KWH[target]
        for target in targets
    }


def select_pooled_alpha(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    n_estimators: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select one common quantile using only 2023 Jan-Sep -> Q4."""
    split = pd.Timestamp("2023-10-01 00:00:00")
    train = features.loc[features.index < split]
    query = features.loc[features.index >= split]
    records: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        predictions = fit_pooled_expert(
            train,
            labels,
            query,
            targets,
            alpha=alpha,
            seed=13,
            n_estimators=n_estimators,
        )
        group_metrics = {
            target: evaluate_group(
                labels[target].reindex(query.index).to_numpy(dtype=float),
                predictions[target],
                CAPACITY_KWH[target],
            ).to_dict()
            for target in targets
        }
        records.append(
            {
                "alpha": float(alpha),
                "macro_score": float(
                    np.mean(
                        [
                            group_metrics[target]["score"]
                            for target in targets
                        ]
                    )
                ),
                "groups": group_metrics,
            }
        )
    selected = max(
        records,
        key=lambda row: (row["macro_score"], -row["alpha"]),
    )
    return float(selected["alpha"]), records


def combine_year_features(
    features: dict[str, pd.DataFrame],
    years: tuple[str, ...],
) -> pd.DataFrame:
    """Concatenate causal training years with an exact feature contract."""
    if not years:
        raise ValueError("at least one training year is required")
    missing = [year for year in years if year not in features]
    if missing:
        raise ValueError(f"training features are missing years: {missing}")
    reference_columns = features[years[0]].columns
    for year in years[1:]:
        if not features[year].columns.equals(reference_columns):
            raise ValueError(
                f"training feature columns differ for {years[0]} and {year}"
            )
    combined = pd.concat([features[year] for year in years], axis=0)
    combined = combined.sort_index()
    if combined.index.duplicated().any():
        raise ValueError("training years contain duplicate timestamps")
    if combined.isna().any().any():
        raise ValueError("combined training features are incomplete")
    return combined.astype("float32")


def select_plateau_mean_weight(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Average the Q1 near-best weight plateau on the explicit grid."""
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise RuntimeError("no pooled blend weight improved every Q1 component")
    best_score = max(record["delta"]["score"] for record in eligible)
    near_best = [
        record
        for record in eligible
        if record["delta"]["score"]
        >= best_score - PARSIMONY_SCORE_TOLERANCE
    ]
    plateau_mean = float(
        np.mean([float(record["weight"]) for record in near_best])
    )
    selected = min(
        eligible,
        key=lambda record: (
            abs(float(record["weight"]) - plateau_mean),
            float(record["weight"]),
        ),
    )
    return selected, {
        "best_q1_score": float(best_score),
        "score_tolerance": float(PARSIMONY_SCORE_TOLERANCE),
        "near_best_weights": [
            float(record["weight"]) for record in near_best
        ],
        "plateau_mean_weight": plateau_mean,
        "selected_grid_weight": float(selected["weight"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    targets = parse_targets(args.targets)
    unsupported = set(targets).difference(SUPPORTED_TARGETS)
    if unsupported:
        raise ValueError(f"unsupported pooled targets: {sorted(unsupported)}")
    weight_grid = tuple(
        float(value.strip())
        for value in args.weights.split(",")
        if value.strip()
    )
    if (
        not weight_grid
        or any(weight <= 0.0 or weight > 0.20 for weight in weight_grid)
        or tuple(sorted(set(weight_grid))) != weight_grid
    ):
        raise ValueError(
            "weights must be unique, increasing, and within (0, 0.20]"
        )
    validation_only = bool(args.validation_only)
    training_years = ("2022", "2023") if args.context_2022 else ("2023",)
    years = training_years + (
        ("2024",) if validation_only else ("2024", "2025")
    )
    paths: dict[str, Path] = {}
    manifest_validation: dict[str, Any] = {}
    features: dict[str, pd.DataFrame] = {}
    issues: dict[str, pd.Series] = {}
    for year in years:
        primary = _rooted(getattr(args, f"context_{year}"))
        primary_manifest = _rooted(getattr(args, f"manifest_{year}"))
        extra = tuple(
            _rooted(value)
            for value in getattr(args, f"extra_context_{year}")
        )
        extra_manifest = tuple(
            _rooted(value)
            for value in getattr(args, f"extra_manifest_{year}")
        )
        if len(extra) != len(extra_manifest):
            raise ValueError(f"{year} extra context/manifest counts differ")
        paths[f"context_{year}"] = primary
        paths[f"manifest_{year}"] = primary_manifest
        manifest_validation[year] = validate_external_data_manifest(
            primary_manifest,
            ROOT,
        )
        for position, (context_path, manifest_path) in enumerate(
            zip(extra, extra_manifest),
            start=1,
        ):
            paths[f"extra_context_{year}_{position}"] = context_path
            paths[f"extra_manifest_{year}_{position}"] = manifest_path
            manifest_validation[f"{year}_extra_{position}"] = (
                validate_external_data_manifest(manifest_path, ROOT)
            )
        features[year], issues[year] = load_context_feature_bundle(
            primary,
            extra,
        )
    training_features = combine_year_features(features, training_years)

    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    driver = np.load(_rooted(args.driver), allow_pickle=False)
    kma_oof = np.load(_rooted(args.kma_oof), allow_pickle=False)
    group2_oof = None
    group2_policy = None
    if "kpx_group_2" in targets:
        group2_oof_path = _rooted(args.group2_oof)
        group2_report_path = _rooted(args.group2_overlay_report)
        group2_oof = np.load(group2_oof_path, allow_pickle=False)
        group2_report = json.loads(
            group2_report_path.read_text(encoding="utf-8")
        )
        raw_policy = group2_report["search"]["exploratory_by_group"][
            "kpx_group_2"
        ]["policy"]
        group2_policy = OverlayPolicy(
            **{**raw_policy, "alpha": float(args.group2_alpha)}
        )
        paths["group2_oof"] = group2_oof_path
        paths["group2_overlay_report"] = group2_report_path

    if args.select_alpha_2023_q4:
        selected_alpha, alpha_records = select_pooled_alpha(
            training_features,
            labels,
            targets,
            n_estimators=args.n_estimators,
        )
        alpha_contract = (
            f"selected on pooled {'+'.join(training_years)} through "
            "2023 Sep -> 2023 Q4"
        )
    else:
        selected_alpha = float(args.alpha)
        alpha_records = []
        alpha_contract = "fixed before validation"
    seed_predictions = [
        fit_pooled_expert(
            training_features,
            labels,
            features["2024"],
            targets,
            alpha=selected_alpha,
            seed=seed,
            n_estimators=args.n_estimators,
        )
        for seed in SEEDS
    ]

    validation: dict[str, Any] = {}
    selected_specs: dict[str, dict[str, float]] = {}
    promoted_targets: list[str] = []
    near_stable_targets: list[str] = []
    validation_cache: dict[str, np.ndarray] = {}
    expected_macro_delta = 0.0
    near_stable_macro_delta = 0.0
    for target in targets:
        capacity = CAPACITY_KWH[target]
        index, truth, reference = _validation_surfaces(
            driver,
            kma_oof,
            target,
            group2_oof=group2_oof,
            group2_policy=group2_policy,
        )
        aligned = [
            _align_prediction(
                member[target],
                features["2024"].index,
                index,
                reference,
            )
            for member in seed_predictions
        ]
        expert = np.mean(aligned, axis=0)
        periods = _periods(index)
        weight_rows = evaluate_weights(
            truth,
            reference,
            expert,
            capacity,
            periods["q1"],
            weights=weight_grid,
        )
        eligible = [record for record in weight_rows if record["eligible"]]
        plateau_selection: dict[str, Any] | None = None
        if eligible:
            if args.weight_selection_mode == "plateau_mean":
                selected, plateau_selection = select_plateau_mean_weight(
                    weight_rows
                )
            else:
                selected, _ = select_weight(
                    truth,
                    reference,
                    expert,
                    capacity,
                    periods["q1"],
                    weights=weight_grid,
                )
            q1_eligible = True
        else:
            selected = max(
                weight_rows,
                key=lambda row: (
                    row["delta"]["score"],
                    min(row["delta"].values()),
                    -row["weight"],
                ),
            )
            q1_eligible = False
        weight = float(selected["weight"])
        candidate = apply_bounded_blend(
            reference,
            expert,
            weight=weight,
            capacity=capacity,
        )
        period_deltas = {
            name: metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                rows,
            )
            for name, rows in periods.items()
        }
        seed_rows = []
        for seed, member in zip(SEEDS, aligned):
            seed_candidate = apply_bounded_blend(
                reference,
                member,
                weight=weight,
                capacity=capacity,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "period_deltas": {
                        name: metric_delta(
                            truth,
                            reference,
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
                reference,
                candidate,
                capacity,
                periods["full"] & np.asarray(index.month == month),
            )
            for month in range(1, 13)
        }
        issue = issues["2024"].reindex(index)
        if int(issue.loc[index < VALIDATION_END].isna().sum()):
            raise ValueError("issue times are missing inside validation year")
        issue = issue.fillna(pd.Timestamp("2024-12-31 13:00")).to_numpy()
        bootstrap = evaluate_blocked_rolling(
            truth,
            reference,
            candidate,
            index,
            issue,
            periods["full"] & (truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260726,
        )
        positive_months = int(
            sum(value["score"] > 0.0 for value in monthly.values())
        )
        minimum_seed_split_component = float(
            min(
                value
                for row in seed_rows
                for period in ("q1", "q2", "h2")
                for value in row["period_deltas"][period].values()
            )
        )
        gates = {
            "q1_selection_all_components_positive": q1_eligible,
            "q2_components_positive": min(period_deltas["q2"].values()) > 0.0,
            "h2_components_positive": min(period_deltas["h2"].values()) > 0.0,
            "full_components_positive": min(period_deltas["full"].values()) > 0.0,
            "all_seed_split_components_positive": all(
                min(row["period_deltas"][period].values()) > 0.0
                for row in seed_rows
                for period in ("q1", "q2", "h2")
            ),
            "positive_score_month_fraction_at_least_80pct": (
                positive_months / 12.0 >= 0.80
            ),
            "issue_bootstrap_q05_positive": (
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "issue_bootstrap_positive_fraction_at_least_98pct": (
                bootstrap["issue_block_bootstrap"]["positive_fraction"] >= 0.98
            ),
        }
        promoted = bool(all(gates.values()))
        near_stable = bool(
            not promoted
            and all(
                value
                for name, value in gates.items()
                if name != "all_seed_split_components_positive"
            )
            and minimum_seed_split_component
            >= args.near_stable_seed_component_floor
        )
        if promoted:
            promoted_targets.append(target)
            expected_macro_delta += period_deltas["full"]["score"] / 3.0
        elif near_stable:
            near_stable_targets.append(target)
            near_stable_macro_delta += (
                period_deltas["full"]["score"] / 3.0
            )
        selected_specs[target] = {
            "alpha": selected_alpha,
            "weight": weight,
        }
        validation_cache.update(
            {
                f"{target}__index_ns": index.asi8,
                f"{target}__issue_ns": pd.DatetimeIndex(issue).asi8,
                f"{target}__truth": truth.astype("float32"),
                f"{target}__reference": reference.astype("float32"),
                f"{target}__expert": expert.astype("float32"),
                f"{target}__candidate": candidate.astype("float32"),
            }
        )
        validation[target] = {
            "feature_count": int(training_features.shape[1]),
            "pooled_training_rows": int(
                len(training_features) * len(targets)
            ),
            "training_years": list(training_years),
            "weight_selection_2024_q1": {
                "selected": selected,
                "records": weight_rows,
                "eligible": q1_eligible,
                "mode": args.weight_selection_mode,
                "plateau": plateau_selection,
            },
            "period_deltas": period_deltas,
            "seed_stability": seed_rows,
            "monthly_deltas": monthly,
            "positive_score_months": positive_months,
            "issue_block_validation": bootstrap,
            "gates": gates,
            "minimum_seed_split_component_delta": (
                minimum_seed_split_component
            ),
            "near_stable_seed_component_floor": float(
                args.near_stable_seed_component_floor
            ),
            "promotion": "promoted" if promoted else "rejected",
            "promotion_tier": (
                "strict"
                if promoted
                else "near_stable"
                if near_stable
                else "rejected"
            ),
        }

    cache_path = _rooted(args.output_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **validation_cache)

    candidate_record: dict[str, Any] | None = None
    deployment_targets = list(promoted_targets)
    if args.allow_near_stable:
        deployment_targets.extend(near_stable_targets)
    if deployment_targets and not validation_only:
        production_training_features = combine_year_features(
            features,
            training_years + ("2024",),
        )
        production_predictions = [
            fit_pooled_expert(
                production_training_features,
                labels,
                features["2025"],
                targets,
                alpha=selected_alpha,
                seed=seed,
                n_estimators=args.n_estimators,
            )
            for seed in SEEDS
        ]
        active = pd.read_csv(
            _rooted(args.active_candidate),
            encoding="utf-8-sig",
        )
        output = active.copy()
        for target in deployment_targets:
            expert = np.mean(
                [member[target] for member in production_predictions],
                axis=0,
            )
            output[target] = apply_bounded_blend(
                active[target].to_numpy(dtype=float),
                expert,
                weight=selected_specs[target]["weight"],
                capacity=CAPACITY_KWH[target],
            )
        output_path = _rooted(args.output_submission)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        audit = CandidateValidator(load_config(ROOT)).audit(output_path)
        if not audit.valid:
            raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")
        candidate_record = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
            "deployed_targets": deployment_targets,
            "deployment_tier": (
                "near_stable_exploratory"
                if any(
                    target in near_stable_targets
                    for target in deployment_targets
                )
                else "strict"
            ),
        }

    report = {
        "family": "kma_pooled_group_quantile_blend",
        "method": (
            "capacity-normalized pooled groups with target one-hot columns; "
            "fixed quantile and Q1-only bounded blend selection"
        ),
        "sources": {
            key: path.relative_to(ROOT).as_posix()
            for key, path in paths.items()
        },
        "manifest_validation": manifest_validation,
        "contract": {
            "targets": list(targets),
            "alpha": selected_alpha,
            "alpha_selection": alpha_contract,
            "validation_training": (
                f"all {'+'.join(training_years)} pooled groups -> 2024"
            ),
            "training_years": list(training_years),
            "weight_selection": (
                f"2024 Q1 only ({args.weight_selection_mode})"
            ),
            "weight_grid": list(weight_grid),
            "confirmation_exposure_disclosure": (
                "plateau-mean follow-up was proposed after inspecting the "
                "earlier 2024 confirmation report on 2026-07-28"
                if args.weight_selection_mode == "plateau_mean"
                else None
            ),
            "confirmation": "Q2, H2, seeds, months, issue-block bootstrap",
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "near_stable_seed_component_floor": float(
                args.near_stable_seed_component_floor
            ),
            "near_stable_deployment_enabled": bool(args.allow_near_stable),
        },
        "alpha_selection_records": alpha_records,
        "selected_specs": selected_specs,
        "promoted_targets": promoted_targets,
        "near_stable_targets": near_stable_targets,
        "deployment_targets": deployment_targets,
        "validation": validation,
        "expected_macro_score_delta_if_2024_transfers": float(
            expected_macro_delta
        ),
        "projected_public_score_if_local_delta_transfers": float(
            args.active_public_score + expected_macro_delta
        ),
        "expected_macro_score_delta_with_near_stable": float(
            expected_macro_delta + near_stable_macro_delta
        ),
        "projected_public_score_with_near_stable": float(
            args.active_public_score
            + expected_macro_delta
            + near_stable_macro_delta
        ),
        "candidate": candidate_record,
        "validation_cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(cache_path),
            "arrays": sorted(validation_cache),
        },
    }
    report_path = _rooted(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    parser.add_argument(
        "--kma-oof",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--group2-oof",
        default="artifacts_final/lineage/base_v2_group2_complete_oof.npz",
    )
    parser.add_argument(
        "--group2-overlay-report",
        default=(
            "artifacts_final/base_v2/"
            "kma_incumbent_local_overlay_20260725.json"
        ),
    )
    parser.add_argument("--group2-alpha", type=float, default=0.2375)
    parser.add_argument("--context-2022")
    parser.add_argument("--manifest-2022")
    parser.add_argument(
        "--extra-context-2022",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--extra-manifest-2022",
        action="append",
        default=[],
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
            f"--manifest-{year}",
            default=(
                "artifacts_final/external_weather/"
                f"kma_um_regional_context_{year}/manifest.json"
            ),
        )
        parser.add_argument(
            f"--extra-context-{year}",
            action="append",
            default=[],
        )
        parser.add_argument(
            f"--extra-manifest-{year}",
            action="append",
            default=[],
        )
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--select-alpha-2023-q4",
        action="store_true",
        help="select one pooled quantile on 2023 Jan-Sep -> Q4 only",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--weights",
        default="0.025,0.05,0.075,0.10,0.15,0.20",
        help=(
            "comma-separated Q1-only bounded blend grid; use a denser local "
            "grid to audit weight discretization without opening Q2/H2"
        ),
    )
    parser.add_argument(
        "--weight-selection-mode",
        choices=("parsimony", "plateau_mean"),
        default="parsimony",
        help=(
            "parsimony chooses the smallest Q1 near-best weight; "
            "plateau_mean averages the Q1 near-best grid plateau"
        ),
    )
    parser.add_argument(
        "--allow-near-stable",
        action="store_true",
        help=(
            "production-only exploratory deployment for targets whose sole "
            "failure is a seed component above the explicit tolerance"
        ),
    )
    parser.add_argument(
        "--near-stable-seed-component-floor",
        type=float,
        default=-0.000025,
    )
    parser.add_argument(
        "--active-candidate",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument("--active-public-score", type=float, default=0.6440998116)
    parser.add_argument(
        "--output-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_pooled_group_quantile_20260726.csv"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_pooled_group_quantile_20260726.json"
        ),
    )
    parser.add_argument(
        "--output-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_pooled_group_quantile_20260726.npz"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "selected_specs": report["selected_specs"],
                "promoted_targets": report["promoted_targets"],
                "near_stable_targets": report["near_stable_targets"],
                "deployment_targets": report["deployment_targets"],
                "expected_macro_score_delta": report[
                    "expected_macro_score_delta_if_2024_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_local_delta_transfers"
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
