"""Audit and blend mechanism-diverse wind OOF surfaces.

The experiment transfers the successful *diverse paradigms around a strong
anchor* idea from the mosquito-trajectory solution.  It freezes group 3 after
repeated public failures, searches only a small convex displacement blend on
2024 Q1, and confirms the locked blend on Q2/H2.  It never writes a submission.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    complementary_subset_stress_all,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
    movement_summary,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
FROZEN_TARGET = "kpx_group_3"
DEFAULT_PRIMARY_CACHE = (
    "artifacts_final/lineage/"
    "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
)
DEFAULT_RESIDUAL_CACHE = (
    "artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz"
)
DEFAULT_GROUP3_CACHE = (
    "artifacts_final/external_weather/"
    "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def align_to_index(
    target_index: pd.DatetimeIndex,
    source_index: pd.DatetimeIndex,
    source_values: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Align a partial OOF vector and fill unavailable rows from the anchor."""
    if source_index.has_duplicates:
        raise ValueError("source OOF index has duplicates")
    source = pd.Series(np.asarray(source_values, dtype=float), index=source_index)
    aligned = source.reindex(target_index)
    available = aligned.notna().to_numpy()
    output = np.asarray(fallback, dtype=float).copy()
    output[available] = aligned.to_numpy(dtype=float, na_value=np.nan)[available]
    if not np.isfinite(output).all():
        raise ValueError("aligned OOF surface is non-finite")
    return output, int(np.sum(available))


def _truth_matches(
    source: np.ndarray,
    truth: np.ndarray,
    available: np.ndarray | None = None,
) -> bool:
    rows = np.ones(len(truth), dtype=bool) if available is None else available
    return bool(
        np.allclose(
            np.asarray(source, dtype=float)[rows],
            np.asarray(truth, dtype=float)[rows],
            rtol=0.0,
            atol=1e-6,
        )
    )


def load_mechanisms(
    index: pd.DatetimeIndex,
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    *,
    driver_cache: Path,
    ldaps_caches: dict[str, Path],
    trajectory_cache: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    mechanisms: dict[str, dict[str, np.ndarray]] = {}
    coverage: dict[str, Any] = {}

    with np.load(driver_cache, allow_pickle=False) as cache:
        for name, key in (
            ("legacy_driver_exact", "exact_base"),
            ("legacy_driver_stack5", "stack5"),
        ):
            surface = {target: active[target].copy() for target in TARGETS}
            counts: dict[str, int] = {}
            for target in TARGETS:
                prefix = f"{target}__"
                source_index = pd.DatetimeIndex(
                    pd.to_datetime(cache[prefix + "valid_index_ns"])
                )
                values, count = align_to_index(
                    index,
                    source_index,
                    cache[prefix + key],
                    active[target],
                )
                source_truth, _ = align_to_index(
                    index,
                    source_index,
                    cache[prefix + "valid_truth"],
                    truth[target],
                )
                if not _truth_matches(source_truth, truth[target]):
                    raise ValueError(f"driver truth mismatch for {target}")
                if target != FROZEN_TARGET:
                    surface[target] = values
                counts[target] = count
            mechanisms[name] = surface
            coverage[name] = counts

    ldaps_surface = {target: active[target].copy() for target in TARGETS}
    ldaps_counts: dict[str, int] = {}
    for target, path in ldaps_caches.items():
        with np.load(path, allow_pickle=False) as cache:
            prefix = f"{target}__"
            source_index = pd.DatetimeIndex(
                pd.to_datetime(cache[prefix + "index_ns"])
            )
            values, count = align_to_index(
                index,
                source_index,
                cache[prefix + "candidate"],
                active[target],
            )
            source_truth, _ = align_to_index(
                index,
                source_index,
                cache[prefix + "truth"],
                truth[target],
            )
            if not _truth_matches(source_truth, truth[target]):
                raise ValueError(f"LDAPS truth mismatch for {target}")
            if target != FROZEN_TARGET:
                ldaps_surface[target] = values
            ldaps_counts[target] = count
    mechanisms["ldaps_multiresolution"] = ldaps_surface
    coverage["ldaps_multiresolution"] = ldaps_counts

    with np.load(trajectory_cache, allow_pickle=False) as cache:
        trajectory = {target: active[target].copy() for target in TARGETS}
        counts = {}
        for target in TARGETS:
            prefix = f"{target}__"
            source_index = pd.DatetimeIndex(
                pd.to_datetime(cache[prefix + "index_ns"])
            )
            values, count = align_to_index(
                index,
                source_index,
                cache[prefix + "candidate"],
                active[target],
            )
            source_truth, _ = align_to_index(
                index,
                source_index,
                cache[prefix + "truth"],
                truth[target],
            )
            if not _truth_matches(source_truth, truth[target]):
                raise ValueError(f"trajectory truth mismatch for {target}")
            if target != FROZEN_TARGET:
                trajectory[target] = values
            counts[target] = count
        mechanisms["issue_trajectory_tcn"] = trajectory
        coverage["issue_trajectory_tcn"] = counts

    return mechanisms, coverage


def period_rows(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    months = interval_month(index)
    return {
        "q1": months <= 3,
        "q2": (months >= 4) & (months <= 6),
        "h2": months >= 7,
        "full": np.ones(len(index), dtype=bool),
    }


def compose_displacement_blend(
    active: dict[str, np.ndarray],
    mechanisms: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    *,
    total_alpha: float,
) -> dict[str, np.ndarray]:
    if not 0.0 <= total_alpha <= 1.0:
        raise ValueError("total alpha must lie in [0, 1]")
    if not weights:
        raise ValueError("at least one mechanism weight is required")
    if any(name not in mechanisms for name in weights):
        raise ValueError("unknown mechanism in blend")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("mechanism weights must be nonnegative")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("mechanism weights must sum to one")

    output: dict[str, np.ndarray] = {}
    for target in TARGETS:
        base = np.asarray(active[target], dtype=float)
        displacement = sum(
            float(weight)
            * (np.asarray(mechanisms[name][target], dtype=float) - base)
            for name, weight in weights.items()
        )
        output[target] = np.clip(
            base + float(total_alpha) * displacement,
            0.0,
            CAPACITY_KWH[target],
        )
    if not np.array_equal(output[FROZEN_TARGET], active[FROZEN_TARGET]):
        raise AssertionError("group 3 changed despite the frozen-target contract")
    return output


def _mixtures(names: tuple[str, ...]) -> list[dict[str, float]]:
    mixtures: list[dict[str, float]] = []
    for name in names:
        mixtures.append({name: 1.0})
    for pair in itertools.combinations(names, 2):
        for left in (0.25, 0.50, 0.75):
            mixtures.append({pair[0]: left, pair[1]: 1.0 - left})
    for triple in itertools.combinations(names, 3):
        patterns = (
            (1 / 3, 1 / 3, 1 / 3),
            (0.50, 0.25, 0.25),
            (0.25, 0.50, 0.25),
            (0.25, 0.25, 0.50),
        )
        for pattern in patterns:
            mixtures.append(dict(zip(triple, pattern, strict=True)))
    return mixtures


def q1_q2_stability_score(record: dict[str, Any]) -> float:
    """Return the worst score delta exposed before the locked H2 check."""
    q2 = record["q2_delta_if_q1_eligible"]
    q2_months = record["q2_monthly_deltas_if_q1_eligible"]
    if q2 is None or q2_months is None:
        return float("-inf")
    return float(
        min(
            record["q1_delta"]["score"],
            q2["score"],
            *(value["score"] for value in record["q1_monthly_deltas"].values()),
            *(value["score"] for value in q2_months.values()),
        )
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def diversity_matrix(
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    mechanisms: dict[str, dict[str, np.ndarray]],
    rows: np.ndarray,
) -> dict[str, Any]:
    names = tuple(mechanisms)
    movements: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    for name, surface in mechanisms.items():
        movements[name] = np.concatenate(
            [
                (surface[target][rows] - active[target][rows])
                / CAPACITY_KWH[target]
                for target in TARGETS
            ]
        )
        errors[name] = np.concatenate(
            [
                (truth[target][rows] - surface[target][rows])
                / CAPACITY_KWH[target]
                for target in TARGETS
            ]
        )
    return {
        "movement_correlation": {
            left: {
                right: _correlation(movements[left], movements[right])
                for right in names
            }
            for left in names
        },
        "error_correlation": {
            left: {
                right: _correlation(errors[left], errors[right])
                for right in names
            }
            for left in names
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    baselines, truths, index, issues = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    active = {
        target: baselines[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    truth = {
        target: truths[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    mechanisms, coverage = load_mechanisms(
        index,
        truth,
        active,
        driver_cache=_rooted(args.driver_cache),
        ldaps_caches={
            target: _rooted(getattr(args, f"ldaps_{target[-1]}_cache"))
            for target in TARGETS
        },
        trajectory_cache=_rooted(args.trajectory_cache),
    )
    rows = period_rows(index)
    months = interval_month(index)
    member_deltas = {
        name: {
            period: metric_delta(truth, active, surface, mask)
            for period, mask in rows.items()
        }
        for name, surface in mechanisms.items()
    }

    alpha_grid = tuple(
        float(value.strip())
        for value in args.alphas.split(",")
        if value.strip()
    )
    records: list[dict[str, Any]] = []
    for weights in _mixtures(tuple(mechanisms)):
        for alpha in alpha_grid:
            candidate = compose_displacement_blend(
                active, mechanisms, weights, total_alpha=alpha
            )
            q1 = metric_delta(truth, active, candidate, rows["q1"])
            q1_months = {
                str(month): metric_delta(
                    truth,
                    active,
                    candidate,
                    rows["q1"] & (months == month),
                )
                for month in (1, 2, 3)
            }
            movement = movement_summary(active, candidate)
            eligible = bool(
                all(q1[component] > 0.0 for component in COMPONENTS)
                and all(value["score"] >= 0.0 for value in q1_months.values())
                and movement["p95_ratio"] <= args.maximum_p95_ratio
            )
            q2 = None
            q2_months = None
            q2_eligible = False
            if eligible:
                q2 = metric_delta(truth, active, candidate, rows["q2"])
                q2_months = {
                    str(month): metric_delta(
                        truth,
                        active,
                        candidate,
                        rows["q2"] & (months == month),
                    )
                    for month in (4, 5, 6)
                }
                q2_eligible = bool(
                    all(q2[component] > 0.0 for component in COMPONENTS)
                    and all(
                        value["score"] >= 0.0
                        for value in q2_months.values()
                    )
                )
            posthoc_periods = {
                "q1": q1,
                "q2": (
                    q2
                    if q2 is not None
                    else metric_delta(
                        truth, active, candidate, rows["q2"]
                    )
                ),
                "h2": metric_delta(
                    truth, active, candidate, rows["h2"]
                ),
                "full": metric_delta(
                    truth, active, candidate, rows["full"]
                ),
            }
            posthoc_minimum_component = min(
                posthoc_periods[period][component]
                for period in ("q1", "q2", "h2", "full")
                for component in COMPONENTS
            )
            records.append(
                {
                    "weights": weights,
                    "total_alpha": alpha,
                    "q1_delta": q1,
                    "q1_monthly_deltas": q1_months,
                    "movement": movement,
                    "eligible": eligible,
                    "q2_delta_if_q1_eligible": q2,
                    "q2_monthly_deltas_if_q1_eligible": q2_months,
                    "q2_transfer_eligible": q2_eligible,
                    "posthoc_period_deltas": posthoc_periods,
                    "posthoc_minimum_period_component": (
                        posthoc_minimum_component
                    ),
                }
            )
    eligible_records = [record for record in records if record["eligible"]]
    if not eligible_records:
        selected = None
        selected_candidate = active
    else:
        best_gain = max(record["q1_delta"]["score"] for record in eligible_records)
        plateau = [
            record
            for record in eligible_records
            if record["q1_delta"]["score"] >= 0.95 * best_gain
        ]
        selected = min(
            plateau,
            key=lambda record: (
                record["total_alpha"],
                len(record["weights"]),
                -record["q1_delta"]["score"],
            ),
        )
        selected_candidate = compose_displacement_blend(
            active,
            mechanisms,
            selected["weights"],
            total_alpha=selected["total_alpha"],
        )

    period_deltas = {
        name: metric_delta(truth, active, selected_candidate, mask)
        for name, mask in rows.items()
    }
    monthly_deltas = {
        str(month): metric_delta(
            truth, active, selected_candidate, months == month
        )
        for month in range(1, 13)
    }
    movement = movement_summary(active, selected_candidate)
    pre_stress = bool(
        selected is not None
        and all(
            period_deltas[period][component] > 0.0
            for period in ("q2", "h2", "full")
            for component in COMPONENTS
        )
        and all(value["score"] >= 0.0 for value in monthly_deltas.values())
    )
    iid = None
    stratified = None
    bootstrap = None
    if pre_stress:
        iid = complementary_subset_stress_all(
            truth,
            active,
            selected_candidate,
            index,
            rows["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed,
            stratify_month=False,
        )
        stratified = complementary_subset_stress_all(
            truth,
            active,
            selected_candidate,
            index,
            rows["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed + 1,
            stratify_month=True,
        )
        bootstrap = issue_block_bootstrap_all(
            truth,
            active,
            selected_candidate,
            index,
            issues,
            rows["h2"],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 2,
        )

    def subset_pass(result: dict[str, Any] | None) -> bool:
        return bool(
            result is not None
            and all(
                result[split][component]["q05"] >= 0.0
                for split in ("public", "private")
                for component in COMPONENTS
            )
        )

    bootstrap_pass = bool(
        bootstrap is not None
        and all(
            bootstrap["summary"][component]["q05"] >= 0.0
            for component in COMPONENTS
        )
    )
    promotion = bool(
        pre_stress
        and subset_pass(iid)
        and subset_pass(stratified)
        and bootstrap_pass
    )
    q2_transfer_records = [
        record for record in eligible_records if record["q2_transfer_eligible"]
    ]
    if q2_transfer_records:
        best_q2_gain = max(
            record["q2_delta_if_q1_eligible"]["score"]
            for record in q2_transfer_records
        )
        q2_plateau = [
            record
            for record in q2_transfer_records
            if record["q2_delta_if_q1_eligible"]["score"]
            >= 0.95 * best_q2_gain
        ]
        q2_selected = min(
            q2_plateau,
            key=lambda record: (
                record["total_alpha"],
                len(record["weights"]),
                -record["q2_delta_if_q1_eligible"]["score"],
            ),
        )
        q2_candidate = compose_displacement_blend(
            active,
            mechanisms,
            q2_selected["weights"],
            total_alpha=q2_selected["total_alpha"],
        )
    else:
        q2_selected = None
        q2_candidate = active
    q2_followup_periods = {
        name: metric_delta(truth, active, q2_candidate, mask)
        for name, mask in rows.items()
    }
    q2_followup_months = {
        str(month): metric_delta(truth, active, q2_candidate, months == month)
        for month in range(1, 13)
    }
    q2_followup_pre_stress = bool(
        q2_selected is not None
        and all(
            q2_followup_periods[period][component] > 0.0
            for period in ("h2", "full")
            for component in COMPONENTS
        )
        and all(
            value["score"] >= 0.0 for value in q2_followup_months.values()
        )
    )
    q2_iid = None
    q2_stratified = None
    q2_bootstrap = None
    if q2_followup_pre_stress:
        q2_iid = complementary_subset_stress_all(
            truth,
            active,
            q2_candidate,
            index,
            rows["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed + 10,
            stratify_month=False,
        )
        q2_stratified = complementary_subset_stress_all(
            truth,
            active,
            q2_candidate,
            index,
            rows["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed + 11,
            stratify_month=True,
        )
        q2_bootstrap = issue_block_bootstrap_all(
            truth,
            active,
            q2_candidate,
            index,
            issues,
            rows["h2"],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 12,
        )
    q2_bootstrap_pass = bool(
        q2_bootstrap is not None
        and all(
            q2_bootstrap["summary"][component]["q05"] >= 0.0
            for component in COMPONENTS
        )
    )
    q2_followup_promotion = bool(
        q2_followup_pre_stress
        and subset_pass(q2_iid)
        and subset_pass(q2_stratified)
        and q2_bootstrap_pass
    )
    if q2_transfer_records:
        robust_q2_selected = max(
            q2_transfer_records,
            key=lambda record: (
                q1_q2_stability_score(record),
                -record["total_alpha"],
                -len(record["weights"]),
                record["q2_delta_if_q1_eligible"]["score"],
            ),
        )
        robust_q2_candidate = compose_displacement_blend(
            active,
            mechanisms,
            robust_q2_selected["weights"],
            total_alpha=robust_q2_selected["total_alpha"],
        )
    else:
        robust_q2_selected = None
        robust_q2_candidate = active
    robust_q2_periods = {
        name: metric_delta(truth, active, robust_q2_candidate, mask)
        for name, mask in rows.items()
    }
    robust_q2_months = {
        str(month): metric_delta(
            truth, active, robust_q2_candidate, months == month
        )
        for month in range(1, 13)
    }
    robust_q2_confirmation_pass = bool(
        robust_q2_selected is not None
        and all(
            robust_q2_periods[period][component] > 0.0
            for period in ("h2", "full")
            for component in COMPONENTS
        )
        and all(
            value["score"] >= 0.0 for value in robust_q2_months.values()
        )
    )
    top_records = sorted(
        records,
        key=lambda record: record["q1_delta"]["score"],
        reverse=True,
    )[: args.report_top_records]
    maximin_record = max(
        records,
        key=lambda record: record["posthoc_minimum_period_component"],
    )
    nonnegative_posthoc_records = [
        record
        for record in records
        if record["posthoc_minimum_period_component"] >= 0.0
    ]
    return {
        "family": "mechanism_diversity_anchor_blend",
        "method": (
            "small convex blend of frozen tabular, spatial-physical, and "
            "trajectory displacements around the exact active anchor"
        ),
        "contract": {
            "selection": "Q1 only; 95% score plateau then minimum alpha/paradigms",
            "confirmation": "locked Q2, H2, months, subset and issue-block stress",
            "group3_frozen_after_public_failures": True,
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "submission_writer": False,
            "confirmation_exposure": "research-only; 2024 H2 was previously inspected",
        },
        "coverage": coverage,
        "member_period_deltas": member_deltas,
        "diversity": diversity_matrix(
            truth, active, mechanisms, rows["full"]
        ),
        "search": {
            "mechanisms": list(mechanisms),
            "alpha_grid": list(alpha_grid),
            "records": len(records),
            "eligible_q1_records": len(eligible_records),
            "eligible_q1_to_q2_transfer_records": len(q2_transfer_records),
            "top_q1_records": top_records,
            "posthoc_diagnostic_only": {
                "selection_warning": (
                    "uses Q1, Q2 and H2 jointly; cannot promote a candidate"
                ),
                "all_period_component_nonnegative_records": len(
                    nonnegative_posthoc_records
                ),
                "maximin_record": maximin_record,
            },
        },
        "selected": selected,
        "locked_evaluation": {
            "period_deltas": period_deltas,
            "monthly_deltas": monthly_deltas,
            "movement": movement,
            "pre_stress_pass": pre_stress,
            "iid_complementary_40_60": iid,
            "month_stratified_complementary_40_60": stratified,
            "h2_issue_block_bootstrap": bootstrap,
            "promotion_eligible": promotion,
        },
        "q2_transfer_followup": {
            "contract": (
                "Q1 eligibility screen, Q2 selection, locked H2 confirmation"
            ),
            "selected": q2_selected,
            "period_deltas": q2_followup_periods,
            "monthly_deltas": q2_followup_months,
            "movement": movement_summary(active, q2_candidate),
            "pre_stress_pass": q2_followup_pre_stress,
            "iid_complementary_40_60": q2_iid,
            "month_stratified_complementary_40_60": q2_stratified,
            "h2_issue_block_bootstrap": q2_bootstrap,
            "promotion_eligible": q2_followup_promotion,
        },
        "robust_q1_q2_followup": {
            "contract": (
                "Q1 component/month eligibility, Q2 component/month eligibility, "
                "then maximize the worst Jan-June score delta before one locked "
                "H2 confirmation"
            ),
            "selection_uses_h2": False,
            "selected": robust_q2_selected,
            "q1_q2_stability_score": (
                q1_q2_stability_score(robust_q2_selected)
                if robust_q2_selected is not None
                else None
            ),
            "period_deltas": robust_q2_periods,
            "monthly_deltas": robust_q2_months,
            "movement": movement_summary(active, robust_q2_candidate),
            "locked_h2_confirmation_pass": robust_q2_confirmation_pass,
            "promotion_eligible": False,
            "promotion_blocker": (
                "2024 H2 had already been inspected before this rule was added; "
                "the result is a retrospective hypothesis that requires a new "
                "independent period or a controlled public probe"
            ),
        },
        "verdict": (
            "q1_locked_promoted"
            if promotion
            else "q2_transfer_followup_promoted"
            if q2_followup_promotion
            else "rejected_fail_closed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-cache", default=DEFAULT_PRIMARY_CACHE)
    parser.add_argument("--residual-cache", default=DEFAULT_RESIDUAL_CACHE)
    parser.add_argument("--group3-cache", default=DEFAULT_GROUP3_CACHE)
    parser.add_argument(
        "--driver-cache",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    for target in TARGETS:
        parser.add_argument(
            f"--ldaps-{target[-1]}-cache",
            default=(
                "artifacts_final/lineage/"
                f"ldaps_wind_multires_g{target[-1]}_20260726.npz"
            ),
        )
    parser.add_argument(
        "--trajectory-cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_validation_20260727.npz"
        ),
    )
    parser.add_argument(
        "--alphas", default="0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20"
    )
    parser.add_argument("--maximum-p95-ratio", type=float, default=0.015)
    parser.add_argument("--subset-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--report-top-records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "mechanism_diversity_blend_audit_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    output = _rooted(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "locked_period_deltas": report["locked_evaluation"][
                    "period_deltas"
                ],
                "promotion_eligible": report["locked_evaluation"][
                    "promotion_eligible"
                ],
                "q2_transfer_selected": report["q2_transfer_followup"][
                    "selected"
                ],
                "q2_transfer_period_deltas": report[
                    "q2_transfer_followup"
                ]["period_deltas"],
                "q2_transfer_promotion_eligible": report[
                    "q2_transfer_followup"
                ]["promotion_eligible"],
                "verdict": report["verdict"],
                "output_report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
