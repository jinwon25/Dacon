"""Constrained audit of public-positive factors plus a tiny stabilizer.

The public incumbent already contains two factors whose isolated public
effects were positive: the group-1 residual stack and the group-2 pooled
weather model.  This audit varies only those two factor weights, freezes
group 3, and adds a very small equal displacement from the legacy exact
driver and issue-trajectory TCN.  The latter pair was the only mechanism
blend with non-negative Q1, Q2, H2 and full-period component deltas in the
retrospective diversity audit.

This is deliberately a validation-only, retrospective experiment.  It does
not write a submission and it labels H2 as previously exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)
from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    complementary_subset_stress_all,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
    movement_summary,
)
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.mechanism_diversity_blend_audit import (
    DEFAULT_GROUP3_CACHE,
    DEFAULT_PRIMARY_CACHE,
    DEFAULT_RESIDUAL_CACHE,
    load_mechanisms,
    period_rows,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
FROZEN_TARGET = "kpx_group_3"


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _float_grid(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not output:
        raise ValueError("grid must not be empty")
    return output


def compose_candidate(
    active: dict[str, np.ndarray],
    group1_surface: np.ndarray,
    group2_surface: np.ndarray,
    exact: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    *,
    stabilizer_alpha: float,
    exact_share: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compose independent factor displacements around the active anchor."""
    if stabilizer_alpha < 0.0:
        raise ValueError("stabilizer alpha must be nonnegative")
    if not 0.0 <= exact_share <= 1.0:
        raise ValueError("exact share must lie in [0, 1]")
    output = {
        target: np.asarray(active[target], dtype=float).copy()
        for target in TARGETS
    }
    output["kpx_group_1"] = np.asarray(group1_surface, dtype=float).copy()
    output["kpx_group_2"] = np.asarray(group2_surface, dtype=float).copy()
    for target in ("kpx_group_1", "kpx_group_2"):
        anchor = np.asarray(active[target], dtype=float)
        stabilizer = exact_share * (
            np.asarray(exact[target], dtype=float) - anchor
        ) + (1.0 - exact_share) * (
            np.asarray(trajectory[target], dtype=float) - anchor
        )
        output[target] = np.clip(
            output[target] + stabilizer_alpha * stabilizer,
            0.0,
            CAPACITY_KWH[target],
        )
    if not np.array_equal(output[FROZEN_TARGET], active[FROZEN_TARGET]):
        raise AssertionError("group 3 changed despite the frozen-target contract")
    return output


def _record(
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    periods: dict[str, np.ndarray],
    *,
    group1_weight: float,
    group2_weight: float,
    stabilizer_alpha: float,
) -> dict[str, Any]:
    months = interval_month(index)
    period_deltas = {
        name: metric_delta(truth, active, candidate, rows)
        for name, rows in periods.items()
    }
    monthly = {
        str(month): metric_delta(truth, active, candidate, months == month)
        for month in range(1, 13)
    }
    period_components = [
        period_deltas[name][component]
        for name in ("q1", "q2", "h2", "full")
        for component in COMPONENTS
    ]
    return {
        "group1_residual_weight": group1_weight,
        "group2_pooled_weight": group2_weight,
        "stabilizer_alpha": stabilizer_alpha,
        "period_deltas": period_deltas,
        "minimum_period_component": float(min(period_components)),
        "all_period_components_nonnegative": bool(
            min(period_components) >= 0.0
        ),
        "monthly_score_deltas": {
            month: value["score"] for month, value in monthly.items()
        },
        "positive_score_months": int(
            sum(value["score"] > 0.0 for value in monthly.values())
        ),
        "worst_month_score_delta": float(
            min(value["score"] for value in monthly.values())
        ),
        "movement": movement_summary(active, candidate),
    }


def _subset_pass(result: dict[str, Any]) -> bool:
    return bool(
        all(
            result[split][component]["q05"] >= 0.0
            for split in ("public", "private")
            for component in COMPONENTS
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = _rooted(args.primary_cache)
    residual_path = _rooted(args.residual_cache)
    group3_path = _rooted(args.group3_cache)
    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        primary_path, residual_path, group3_path
    )
    active = {
        target: baselines[target].to_numpy(dtype=float) for target in TARGETS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float) for target in TARGETS
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
    exact = mechanisms["legacy_driver_exact"]
    trajectory = mechanisms["issue_trajectory_tcn"]

    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        residual_path, allow_pickle=False
    ) as residual:
        g1_reference = primary["kpx_group_1__reference"]
        g1_primary = primary["kpx_group_1__candidate"]
        g1_residual = residual["kpx_group_1__candidate"]
        g2_reference = primary["kpx_group_2__reference"]
        g2_expert = primary["kpx_group_2__expert"]

    g1_weights = _float_grid(args.group1_weights)
    g2_weights = _float_grid(args.group2_weights)
    alphas = _float_grid(args.stabilizer_alphas)
    g1_surfaces = {
        weight: apply_capped_residual_stack(
            g1_reference,
            g1_primary,
            g1_residual,
            residual_weight=weight,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        for weight in g1_weights
    }
    g2_surfaces = {
        weight: apply_bounded_blend(
            g2_reference,
            g2_expert,
            weight=weight,
            capacity=CAPACITY_KWH["kpx_group_2"],
        )
        for weight in g2_weights
    }
    periods = period_rows(index)
    records: list[dict[str, Any]] = []
    surfaces: dict[tuple[float, float, float], dict[str, np.ndarray]] = {}
    for group1_weight in g1_weights:
        for group2_weight in g2_weights:
            for alpha in alphas:
                candidate = compose_candidate(
                    active,
                    g1_surfaces[group1_weight],
                    g2_surfaces[group2_weight],
                    exact,
                    trajectory,
                    stabilizer_alpha=alpha,
                    exact_share=args.stabilizer_exact_share,
                )
                key = (group1_weight, group2_weight, alpha)
                surfaces[key] = candidate
                records.append(
                    _record(
                        truth,
                        active,
                        candidate,
                        index,
                        periods,
                        group1_weight=group1_weight,
                        group2_weight=group2_weight,
                        stabilizer_alpha=alpha,
                    )
                )

    eligible = [
        record
        for record in records
        if record["all_period_components_nonnegative"]
        and record["movement"]["p95_ratio"] <= args.maximum_p95_ratio
    ]
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["positive_score_months"],
                record["minimum_period_component"],
                record["period_deltas"]["full"]["score"],
                -record["movement"]["p95_ratio"],
            ),
        )
        if eligible
        else None
    )
    stress = None
    robust_qualified = False
    if selected is not None:
        key = (
            selected["group1_residual_weight"],
            selected["group2_pooled_weight"],
            selected["stabilizer_alpha"],
        )
        candidate = surfaces[key]
        iid = complementary_subset_stress_all(
            truth,
            active,
            candidate,
            index,
            periods["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed,
            stratify_month=False,
        )
        stratified = complementary_subset_stress_all(
            truth,
            active,
            candidate,
            index,
            periods["full"],
            repetitions=args.subset_repetitions,
            seed=args.seed + 1,
            stratify_month=True,
        )
        bootstrap = issue_block_bootstrap_all(
            truth,
            active,
            candidate,
            index,
            issues,
            periods["h2"],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 2,
        )
        gates = {
            "all_period_components_nonnegative": True,
            "at_least_nine_positive_months": bool(
                selected["positive_score_months"] >= 9
            ),
            "iid_complement_q05_nonnegative": _subset_pass(iid),
            "month_stratified_complement_q05_nonnegative": _subset_pass(
                stratified
            ),
            "h2_issue_bootstrap_q05_nonnegative": bool(
                all(
                    bootstrap["summary"][component]["q05"] >= 0.0
                    for component in COMPONENTS
                )
            ),
        }
        robust_qualified = bool(all(gates.values()))
        stress = {
            "iid_complementary_subset": iid,
            "month_stratified_complementary_subset": stratified,
            "h2_issue_block_bootstrap": bootstrap,
            "gates": gates,
            "robust_qualified": robust_qualified,
        }

    report = {
        "schema_version": "public_positive_stabilized_blend.v1",
        "validation_contract": {
            "status": "retrospective hypothesis audit",
            "public_positive_factors_used": True,
            "h2_previously_exposed": True,
            "group3_frozen": True,
            "submission_side_effect": False,
        },
        "configuration": {
            "group1_weights": g1_weights,
            "group2_weights": g2_weights,
            "stabilizer_alphas": alphas,
            "maximum_p95_ratio": args.maximum_p95_ratio,
            "stabilizer": {
                "legacy_driver_exact": args.stabilizer_exact_share,
                "issue_trajectory_tcn": 1.0 - args.stabilizer_exact_share,
            },
        },
        "mechanism_coverage": coverage,
        "searched_candidates": len(records),
        "eligible_candidates": len(eligible),
        "selected": selected,
        "stress": stress,
        "promotion": {
            "robust_qualified": robust_qualified,
            "candidate_written": False,
            "decision": (
                "validation_pass_requires_production_reconstruction"
                if robust_qualified
                else "rejected_or_research_only"
            ),
        },
        "top_eligible": sorted(
            eligible,
            key=lambda record: (
                record["positive_score_months"],
                record["minimum_period_component"],
                record["period_deltas"]["full"]["score"],
            ),
            reverse=True,
        )[: args.report_top_records],
        "top_near_feasible": sorted(
            records,
            key=lambda record: (
                record["minimum_period_component"],
                record["positive_score_months"],
                record["period_deltas"]["full"]["score"],
            ),
            reverse=True,
        )[: args.report_top_records],
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


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
        "--group1-weights", default="0.10,0.1125,0.125,0.1375,0.15,0.1625"
    )
    parser.add_argument(
        "--group2-weights", default="0.07,0.075,0.08,0.0875,0.095,0.10"
    )
    parser.add_argument(
        "--stabilizer-alphas", default="0.0025,0.005,0.0075,0.01,0.0125,0.015"
    )
    parser.add_argument("--stabilizer-exact-share", type=float, default=0.5)
    parser.add_argument("--maximum-p95-ratio", type=float, default=0.01)
    parser.add_argument("--subset-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--report-top-records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_stabilized_blend_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "searched_candidates": report["searched_candidates"],
                "eligible_candidates": report["eligible_candidates"],
                "selected": report["selected"],
                "stress_gates": (
                    None if report["stress"] is None else report["stress"]["gates"]
                ),
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
