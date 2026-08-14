"""Gate additional group-2 dose above the scored 0.1825 anchor.

Submission 1508386 isolates the group-2 expansion at weight 0.1825 while
keeping group 3 frozen.  This diagnostic treats that scored point as the
anchor, searches only the *incremental* group-2 direction, and rejects a
weight when its locked 2024 gain does not transfer across Q2 and H2.

The public score is used only once to calibrate the scale of a local score
delta.  It never selects a weight; local period stability and resampling do.
No submission file is created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.compose_public_positive_multifactor_candidate import (
    OBSERVED_G1_WEIGHT,
)
from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)
from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    complementary_subset_stress_all,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
)
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.mechanism_diversity_blend_audit import period_rows
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
ANCHOR_G2_WEIGHT = 0.1825
CALIBRATION_G2_WEIGHT = 0.095
PUBLIC_CALIBRATION = {
    "lower_submission_id": 1508365,
    "anchor_submission_id": 1508386,
    "lower": {
        "score": 0.6470679857,
        "one_minus_nmae": 0.8759467997,
        "ficr": 0.4181891717,
    },
    "anchor": {
        "score": 0.6474704399,
        "one_minus_nmae": 0.8761125755,
        "ficr": 0.4188283043,
    },
}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_weights(value: str) -> tuple[float, ...]:
    """Parse a comma-delimited strictly increasing weight grid."""
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        raise argparse.ArgumentTypeError("weight grid must not be empty")
    if any(weight <= ANCHOR_G2_WEIGHT for weight in weights):
        raise argparse.ArgumentTypeError(
            f"all weights must exceed the anchor {ANCHOR_G2_WEIGHT}"
        )
    if any(right <= left for left, right in zip(weights, weights[1:])):
        raise argparse.ArgumentTypeError("weights must be strictly increasing")
    return weights


def _public_delta(component: str) -> float:
    return float(
        PUBLIC_CALIBRATION["anchor"][component]
        - PUBLIC_CALIBRATION["lower"][component]
    )


def _subset_q05(result: dict[str, Any], component: str) -> float:
    return float(
        min(
            result[split][component]["q05"]
            for split in ("public", "private")
        )
    )


def _make_surface(
    active: dict[str, np.ndarray],
    primary: Any,
    residual: Any,
    *,
    group2_weight: float,
) -> dict[str, np.ndarray]:
    surface = {target: active[target].copy() for target in TARGETS}
    surface["kpx_group_1"] = apply_capped_residual_stack(
        primary["kpx_group_1__reference"],
        primary["kpx_group_1__candidate"],
        residual["kpx_group_1__candidate"],
        residual_weight=OBSERVED_G1_WEIGHT,
        capacity=CAPACITY_KWH["kpx_group_1"],
        movement_cap_ratio=0.05,
    )
    surface["kpx_group_2"] = apply_bounded_blend(
        primary["kpx_group_2__reference"],
        primary["kpx_group_2__expert"],
        weight=group2_weight,
        capacity=CAPACITY_KWH["kpx_group_2"],
    )
    return surface


def _evaluate_weight(
    truth: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: Any,
) -> dict[str, Any]:
    periods = period_rows(index)
    period_deltas = {
        name: metric_delta(truth, anchor, candidate, rows)
        for name, rows in periods.items()
    }
    months = interval_month(index)
    monthly = {
        str(month): metric_delta(truth, anchor, candidate, months == month)
        for month in range(1, 13)
    }
    return {
        "period_deltas_vs_anchor": period_deltas,
        "monthly_deltas_vs_anchor": monthly,
        "positive_score_months": int(
            sum(record["score"] > 0.0 for record in monthly.values())
        ),
        "pre_stress_gates": {
            "q2_score_nonnegative": bool(period_deltas["q2"]["score"] >= 0.0),
            "h2_score_nonnegative": bool(period_deltas["h2"]["score"] >= 0.0),
            "full_score_positive": bool(period_deltas["full"]["score"] > 0.0),
            "full_ficr_positive": bool(period_deltas["full"]["ficr"] > 0.0),
            "at_least_eight_positive_score_months": bool(
                sum(record["score"] > 0.0 for record in monthly.values()) >= 8
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = _rooted(args.primary_cache)
    residual_path = _rooted(args.residual_cache)
    group3_path = _rooted(args.group3_cache)
    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        primary_path,
        residual_path,
        group3_path,
    )
    active = {
        target: baselines[target].to_numpy(dtype=float) for target in TARGETS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float) for target in TARGETS
    }

    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        residual_path, allow_pickle=False
    ) as residual:
        lower = _make_surface(
            active,
            primary,
            residual,
            group2_weight=CALIBRATION_G2_WEIGHT,
        )
        anchor = _make_surface(
            active,
            primary,
            residual,
            group2_weight=ANCHOR_G2_WEIGHT,
        )
        candidates = {
            weight: _make_surface(
                active,
                primary,
                residual,
                group2_weight=weight,
            )
            for weight in args.weights
        }

    full_rows = np.ones(len(index), dtype=bool)
    local_calibration = metric_delta(truth, lower, anchor, full_rows)
    public_calibration = {
        component: _public_delta(component) for component in COMPONENTS
    }
    direct_score_ratio = (
        public_calibration["score"] / local_calibration["score"]
    )

    evaluations: dict[str, dict[str, Any]] = {}
    for weight, candidate in candidates.items():
        evaluation = _evaluate_weight(truth, anchor, candidate, index)
        full_delta = evaluation["period_deltas_vs_anchor"]["full"]
        evaluation["direct_score_projection"] = {
            "local_incremental_score": float(full_delta["score"]),
            "local_to_public_score_ratio": float(direct_score_ratio),
            "projected_public_score": float(
                PUBLIC_CALIBRATION["anchor"]["score"]
                + full_delta["score"] * direct_score_ratio
            ),
            "warning": (
                "Scale-only projection. Weight selection is governed by locked "
                "local period and resampling gates, not the public projection."
            ),
        }
        evaluations[f"{weight:.4f}"] = evaluation

    locally_best_weight = max(
        args.weights,
        key=lambda weight: evaluations[f"{weight:.4f}"][
            "period_deltas_vs_anchor"
        ]["full"]["score"],
    )
    stress_weight = (
        args.stress_weight if args.stress_weight is not None else locally_best_weight
    )
    if stress_weight not in candidates:
        raise ValueError("stress weight must be included in the weight grid")
    stress_candidate = candidates[stress_weight]
    iid = complementary_subset_stress_all(
        truth,
        anchor,
        stress_candidate,
        index,
        full_rows,
        repetitions=args.subset_repetitions,
        seed=args.seed,
        stratify_month=False,
    )
    stratified = complementary_subset_stress_all(
        truth,
        anchor,
        stress_candidate,
        index,
        full_rows,
        repetitions=args.subset_repetitions,
        seed=args.seed + 1,
        stratify_month=True,
    )
    h2_rows = period_rows(index)["h2"]
    bootstrap = issue_block_bootstrap_all(
        truth,
        anchor,
        stress_candidate,
        index,
        issues,
        h2_rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 2,
    )
    stress = {
        "weight": float(stress_weight),
        "iid_complementary_40_60": iid,
        "month_stratified_complementary_40_60": stratified,
        "h2_issue_block_bootstrap": bootstrap,
        "minimum_q05": {
            component: float(
                min(
                    _subset_q05(iid, component),
                    _subset_q05(stratified, component),
                    bootstrap["summary"][component]["q05"],
                )
            )
            for component in COMPONENTS
        },
    }
    stress["all_component_q05_nonnegative"] = bool(
        all(value >= 0.0 for value in stress["minimum_q05"].values())
    )

    qualified = [
        weight
        for weight in args.weights
        if all(evaluations[f"{weight:.4f}"]["pre_stress_gates"].values())
    ]
    stress_pre_gates = evaluations[f"{stress_weight:.4f}"]["pre_stress_gates"]
    stress_qualified = bool(
        all(stress_pre_gates.values())
        and stress["all_component_q05_nonnegative"]
    )
    result = {
        "schema_version": "public_positive_g2_incremental_gate.v1",
        "contract": {
            "anchor_submission_id": 1508386,
            "anchor_group1_weight": OBSERVED_G1_WEIGHT,
            "anchor_group2_weight": ANCHOR_G2_WEIGHT,
            "group3_frozen": True,
            "public_score_selects_weight": False,
            "submission_created": False,
        },
        "calibration": {
            "public": PUBLIC_CALIBRATION,
            "public_delta": public_calibration,
            "local_delta": local_calibration,
            "direct_score_ratio": float(direct_score_ratio),
            "component_projection_disabled": (
                "The locked local nMAE delta and observed public nMAE delta "
                "have opposite signs, so component-wise extrapolation is invalid."
            ),
        },
        "evaluations": evaluations,
        "locally_best_full_score_weight": float(locally_best_weight),
        "pre_stress_qualified_weights": [float(weight) for weight in qualified],
        "stress": stress,
        "promotion_eligible": stress_qualified,
        "decision": (
            "eligible_incremental_group2_dose"
            if stress_qualified
            else "freeze_group2_at_0.1825"
        ),
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=parse_weights(
            "0.185,0.1875,0.19,0.1925,0.195,0.1975,0.2,0.2025,0.205,"
            "0.2075,0.21,0.2125,0.215,0.2175,0.22,0.2225,0.225,0.2275,"
            "0.23,0.2325,0.235,0.2375,0.24,0.2425,0.245,0.2475,0.25,"
            "0.2525,0.255,0.2575,0.26,0.2625,0.265,0.2675,0.27,0.2725,"
            "0.275,0.2775,0.28,0.2825,0.285,0.2875,0.29,0.2925,0.295,"
            "0.2975,0.3"
        ),
    )
    parser.add_argument("--stress-weight", type=float, default=0.235)
    parser.add_argument("--subset-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
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
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_g2_incremental_gate_20260802.json"
        ),
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "locally_best_full_score_weight": result[
                    "locally_best_full_score_weight"
                ],
                "pre_stress_qualified_weights": result[
                    "pre_stress_qualified_weights"
                ],
                "stress_weight": result["stress"]["weight"],
                "stress_minimum_q05": result["stress"]["minimum_q05"],
                "promotion_eligible": result["promotion_eligible"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
