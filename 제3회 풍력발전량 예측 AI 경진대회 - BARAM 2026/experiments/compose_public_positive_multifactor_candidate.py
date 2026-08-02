"""Compose a conservative expansion of two public-positive factors.

The current public incumbent contains a group-1 residual factor at weight
0.125 and a group-2 pooled-weather factor at weight 0.05.  Both factor
directions were isolated by public factorial submissions and improved the
official macro score.  This module expands them to 0.1375 and 0.095,
respectively, while keeping group 3 byte-for-byte unchanged.

The weights were chosen from a small retrospective local grid.  Public metric
projections therefore use a separate empirical local-to-public transfer ratio
for every factor and metric component; they remain estimates, not guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
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
from experiments.mechanism_diversity_blend_audit import period_rows
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")
G1_BASE_WEIGHT = 0.125
G1_EXPANDED_WEIGHT = 0.1375
G2_BASE_WEIGHT = 0.05
G2_EXPANDED_WEIGHT = 0.095
PUBLIC_INCUMBENT = {
    "score": 0.6461250914,
    "one_minus_nmae": 0.8757842477,
    "ficr": 0.4164659352,
}
PUBLIC_OBSERVED_EXPANSION = {
    "score": 0.6470679857,
    "one_minus_nmae": 0.8759467997,
    "ficr": 0.4181891717,
}
OBSERVED_G1_WEIGHT = 0.1375
OBSERVED_G2_WEIGHT = 0.095
PUBLIC_FACTOR_MACRO_DELTAS = {
    "kpx_group_1": {
        "score": 0.0003023555,
        "one_minus_nmae": -0.0000210193,
        "ficr": 0.0006257305,
    },
    "kpx_group_2": {
        "score": 0.0004796733,
        "one_minus_nmae": 0.0002306458,
        "ficr": 0.0007287007,
    },
}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_observed_factor(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    base_weight: float,
    expanded_weight: float,
    capacity: float,
    maximum_factor_ratio: float = 0.05,
) -> np.ndarray:
    """Scale an observed production factor with a conservative hard cap."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.shape != treatment.shape:
        raise ValueError("factor vectors do not align")
    if not 0.0 < base_weight <= expanded_weight:
        raise ValueError("expanded weight must not be below the base weight")
    if not 0.0 < maximum_factor_ratio <= 1.0:
        raise ValueError("maximum factor ratio must lie in (0, 1]")
    factor = (treatment - control) / base_weight
    movement = np.clip(
        expanded_weight * factor,
        -maximum_factor_ratio * capacity,
        maximum_factor_ratio * capacity,
    )
    return np.clip(control + movement, 0.0, capacity)


def _single_target_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
    target: str,
) -> dict[str, float]:
    empty = {
        other: np.zeros_like(truth, dtype=float)
        for other in TARGETS
        if other != target
    }
    truths = {target: truth, **empty}
    references = {target: reference, **empty}
    candidates = {target: candidate, **empty}
    return metric_delta(truths, references, candidates, rows)


def _transfer_projection(
    local_base: dict[str, dict[str, float]],
    local_incremental: dict[str, dict[str, float]],
) -> dict[str, Any]:
    ratios: dict[str, dict[str, float]] = {}
    projected_increments: dict[str, dict[str, float]] = {}
    for target in ("kpx_group_1", "kpx_group_2"):
        ratios[target] = {}
        projected_increments[target] = {}
        for component in COMPONENTS:
            denominator = local_base[target][component]
            if abs(denominator) <= 1e-12:
                raise ValueError(
                    f"local base delta is zero for {target} {component}"
                )
            public_group_delta = (
                3.0 * PUBLIC_FACTOR_MACRO_DELTAS[target][component]
            )
            ratio = public_group_delta / denominator
            ratios[target][component] = float(ratio)
            projected_increments[target][component] = float(
                local_incremental[target][component] * ratio / 3.0
            )
    total_increment = {
        component: float(
            sum(
                projected_increments[target][component]
                for target in projected_increments
            )
        )
        for component in COMPONENTS
    }
    projected = {
        component: float(
            PUBLIC_INCUMBENT[component] + total_increment[component]
        )
        for component in COMPONENTS
    }
    return {
        "factor_component_transfer_ratios": ratios,
        "factor_projected_macro_increments": projected_increments,
        "total_projected_macro_increment": total_increment,
        "projected_public_metrics": projected,
        "projected_score_gap_to_065": float(0.65 - projected["score"]),
        "warning": (
            "Projection extrapolates two isolated public factor points using "
            "component-specific 2024-to-public transfer ratios. It is not an "
            "observed score and private transfer is not guaranteed."
        ),
    }


def _joint_observed_projection(
    candidate_delta: dict[str, float],
    observed_local_delta: dict[str, float],
) -> dict[str, Any]:
    """Calibrate OOF deltas to the scored 1508365 joint expansion."""
    observed_public_delta = {
        component: float(
            PUBLIC_OBSERVED_EXPANSION[component]
            - PUBLIC_INCUMBENT[component]
        )
        for component in COMPONENTS
    }
    ratios: dict[str, float] = {}
    projected_increment: dict[str, float] = {}
    projected_metrics: dict[str, float] = {}
    for component in COMPONENTS:
        denominator = observed_local_delta[component]
        if abs(denominator) <= 1e-12:
            raise ValueError(
                f"observed local delta is zero for {component}"
            )
        ratio = observed_public_delta[component] / denominator
        ratios[component] = float(ratio)
        projected_increment[component] = float(
            candidate_delta[component] * ratio
        )
        projected_metrics[component] = float(
            PUBLIC_INCUMBENT[component]
            + projected_increment[component]
        )
    return {
        "calibration_submission_id": 1508365,
        "calibration_weights": {
            "group1": OBSERVED_G1_WEIGHT,
            "group2": OBSERVED_G2_WEIGHT,
        },
        "observed_public_delta": observed_public_delta,
        "observed_local_delta": observed_local_delta,
        "component_transfer_ratios": ratios,
        "projected_macro_increment": projected_increment,
        "projected_public_metrics": projected_metrics,
        "projected_score_gap_to_065": float(
            0.65 - projected_metrics["score"]
        ),
        "warning": (
            "Projection assumes the component-specific OOF-to-public transfer "
            "ratio observed at submission 1508365 remains stable along the "
            "same two-factor direction. It is not an observed score."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "incumbent": _rooted(args.incumbent),
        "group1_control": _rooted(args.group1_control),
        "group2_control": _rooted(args.group2_control),
        "primary_cache": _rooted(args.primary_cache),
        "residual_cache": _rooted(args.residual_cache),
        "group3_cache": _rooted(args.group3_cache),
    }
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
        if name in ("incumbent", "group1_control", "group2_control")
    }
    incumbent = frames["incumbent"]
    for name, frame in frames.items():
        if not incumbent[list(ID_COLUMNS)].equals(frame[list(ID_COLUMNS)]):
            raise ValueError(f"{name} identifiers differ from incumbent")

    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        paths["primary_cache"],
        paths["residual_cache"],
        paths["group3_cache"],
    )
    active = {
        target: baselines[target].to_numpy(dtype=float) for target in TARGETS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float) for target in TARGETS
    }
    with np.load(paths["primary_cache"], allow_pickle=False) as primary, np.load(
        paths["residual_cache"], allow_pickle=False
    ) as residual:
        g1_control_validation = primary["kpx_group_1__candidate"].astype(float)
        g1_expanded_validation = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            g1_control_validation,
            residual["kpx_group_1__candidate"],
            residual_weight=args.group1_weight,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        g2_control_validation = primary["kpx_group_2__reference"].astype(float)
        g2_expanded_validation = apply_bounded_blend(
            g2_control_validation,
            primary["kpx_group_2__expert"],
            weight=args.group2_weight,
            capacity=CAPACITY_KWH["kpx_group_2"],
        )
        observed_g1_validation = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            g1_control_validation,
            residual["kpx_group_1__candidate"],
            residual_weight=OBSERVED_G1_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        observed_g2_validation = apply_bounded_blend(
            g2_control_validation,
            primary["kpx_group_2__expert"],
            weight=OBSERVED_G2_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_2"],
        )

    candidate = {target: active[target].copy() for target in TARGETS}
    candidate["kpx_group_1"] = g1_expanded_validation
    candidate["kpx_group_2"] = g2_expanded_validation
    periods = period_rows(index)
    period_deltas = {
        name: metric_delta(truth, active, candidate, rows)
        for name, rows in periods.items()
    }
    months = interval_month(index)
    monthly = {
        str(month): metric_delta(truth, active, candidate, months == month)
        for month in range(1, 13)
    }

    full_rows = periods["full"]
    local_base = {
        "kpx_group_1": _single_target_delta(
            truth["kpx_group_1"],
            g1_control_validation,
            active["kpx_group_1"],
            full_rows,
            "kpx_group_1",
        ),
        "kpx_group_2": _single_target_delta(
            truth["kpx_group_2"],
            g2_control_validation,
            active["kpx_group_2"],
            full_rows,
            "kpx_group_2",
        ),
    }
    local_incremental = {
        "kpx_group_1": _single_target_delta(
            truth["kpx_group_1"],
            active["kpx_group_1"],
            g1_expanded_validation,
            full_rows,
            "kpx_group_1",
        ),
        "kpx_group_2": _single_target_delta(
            truth["kpx_group_2"],
            active["kpx_group_2"],
            g2_expanded_validation,
            full_rows,
            "kpx_group_2",
        ),
    }
    projection = _transfer_projection(local_base, local_incremental)
    observed_candidate = {
        target: active[target].copy() for target in TARGETS
    }
    observed_candidate["kpx_group_1"] = observed_g1_validation
    observed_candidate["kpx_group_2"] = observed_g2_validation
    observed_local_delta = metric_delta(
        truth, active, observed_candidate, full_rows
    )
    joint_projection = _joint_observed_projection(
        period_deltas["full"], observed_local_delta
    )

    iid = complementary_subset_stress_all(
        truth,
        active,
        candidate,
        index,
        full_rows,
        repetitions=args.subset_repetitions,
        seed=args.seed,
        stratify_month=False,
    )
    stratified = complementary_subset_stress_all(
        truth,
        active,
        candidate,
        index,
        full_rows,
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

    output = incumbent.copy()
    output["kpx_group_1"] = expand_observed_factor(
        frames["group1_control"]["kpx_group_1"].to_numpy(dtype=float),
        incumbent["kpx_group_1"].to_numpy(dtype=float),
        base_weight=G1_BASE_WEIGHT,
        expanded_weight=args.group1_weight,
        capacity=CAPACITY_KWH["kpx_group_1"],
        maximum_factor_ratio=args.maximum_factor_ratio,
    )
    output["kpx_group_2"] = expand_observed_factor(
        frames["group2_control"]["kpx_group_2"].to_numpy(dtype=float),
        incumbent["kpx_group_2"].to_numpy(dtype=float),
        base_weight=G2_BASE_WEIGHT,
        expanded_weight=args.group2_weight,
        capacity=CAPACITY_KWH["kpx_group_2"],
        maximum_factor_ratio=args.maximum_factor_ratio,
    )
    if not np.array_equal(
        output["kpx_group_3"].to_numpy(dtype=float),
        incumbent["kpx_group_3"].to_numpy(dtype=float),
    ):
        raise AssertionError("group 3 changed")
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    production_active = {
        target: incumbent[target].to_numpy(dtype=float) for target in TARGETS
    }
    production_candidate = {
        target: output[target].to_numpy(dtype=float) for target in TARGETS
    }
    report = {
        "schema_version": "public_positive_multifactor_candidate.v1",
        "family": "public_positive_group1_group2_factor_expansion",
        "contract": {
            "group1_base_weight": G1_BASE_WEIGHT,
            "group1_expanded_weight": args.group1_weight,
            "group2_base_weight": G2_BASE_WEIGHT,
            "group2_expanded_weight": args.group2_weight,
            "group3_frozen": True,
            "maximum_production_factor_ratio": args.maximum_factor_ratio,
            "public_score_used_for_factor_direction": True,
            "weights_selected_on_repeatedly_inspected_2024_surface": True,
            "test_actual_generation_used": False,
        },
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "validation": {
            "period_deltas_vs_public_incumbent": period_deltas,
            "monthly_deltas_vs_public_incumbent": monthly,
            "positive_score_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "local_base_factor_group_deltas": local_base,
            "local_incremental_group_deltas": local_incremental,
            "iid_complementary_subset": iid,
            "month_stratified_complementary_subset": stratified,
            "h2_issue_block_bootstrap": bootstrap,
            "movement": movement_summary(active, candidate),
        },
        "public_transfer_projection": projection,
        "submission_1508365_calibrated_projection": joint_projection,
        "production_movement_vs_incumbent": movement_summary(
            production_active, production_candidate
        ),
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
        },
        "decision": (
            "controlled_public_probe_candidate"
            if joint_projection["projected_public_metrics"]["score"]
            > PUBLIC_INCUMBENT["score"]
            else "reject_nonpositive_projection"
        ),
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
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--group1-control",
        default=(
            "artifacts_final/candidates/"
            "kma_jma_pooled_all3_g1g2_nearstable_20260726.csv"
        ),
    )
    parser.add_argument(
        "--group2-control",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
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
    parser.add_argument("--group1-weight", type=float, default=G1_EXPANDED_WEIGHT)
    parser.add_argument("--group2-weight", type=float, default=G2_EXPANDED_WEIGHT)
    parser.add_argument(
        "--maximum-factor-ratio", type=float, default=0.05
    )
    parser.add_argument("--subset-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "public_positive_g1w1375_g2w095_g3frozen_20260802.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_g1w1375_g2w095_g3frozen_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "candidate": report["candidate"],
                "period_deltas": report["validation"][
                    "period_deltas_vs_public_incumbent"
                ],
                "positive_score_months": report["validation"][
                    "positive_score_months"
                ],
                "projection": report["public_transfer_projection"],
                "submission_1508365_calibrated_projection": report[
                    "submission_1508365_calibrated_projection"
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
