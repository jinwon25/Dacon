"""Threshold-aware calibration on the aggressive public-positive G2 anchor."""

from __future__ import annotations

import argparse
import itertools
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
    interval_month,
    metric_delta,
    movement_summary,
)
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.mechanism_diversity_blend_audit import (
    DEFAULT_GROUP3_CACHE,
    DEFAULT_PRIMARY_CACHE,
    DEFAULT_RESIDUAL_CACHE,
    period_rows,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
TARGET = "kpx_group_2"
CAPACITY = CAPACITY_KWH[TARGET]


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def apply_policy(
    prediction: np.ndarray,
    policy: dict[str, float | str],
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    kind = str(policy["kind"])
    if kind == "threshold_offset":
        output = prediction.copy()
        action = prediction / CAPACITY >= float(policy["minimum_ratio"])
        output[action] += float(policy["offset"])
    elif kind == "affine":
        output = (
            float(policy["scale"]) * prediction + float(policy["offset"])
        )
    elif kind == "piecewise_offset":
        high = prediction / CAPACITY >= float(policy["breakpoint"])
        output = prediction + np.where(
            high,
            float(policy["high_offset"]),
            float(policy["low_offset"]),
        )
    else:
        raise ValueError(f"unsupported policy: {kind}")
    return np.clip(output, 0.0, CAPACITY)


def _policies() -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for minimum_ratio, offset in itertools.product(
        (0.0, 0.10, 0.20, 0.40, 0.60, 0.80),
        np.arange(-800.0, 800.1, 25.0),
    ):
        if abs(offset) > 1e-12:
            output.append(
                {
                    "kind": "threshold_offset",
                    "minimum_ratio": minimum_ratio,
                    "offset": float(offset),
                }
            )
    for scale, offset in itertools.product(
        np.arange(0.96, 1.0401, 0.002),
        np.arange(-400.0, 400.1, 25.0),
    ):
        if not np.isclose(scale, 1.0) or abs(offset) > 1e-12:
            output.append(
                {"kind": "affine", "scale": float(scale), "offset": float(offset)}
            )
    for breakpoint, low_offset, high_offset in itertools.product(
        (0.20, 0.40, 0.60, 0.80),
        np.arange(-400.0, 400.1, 50.0),
        np.arange(-400.0, 400.1, 50.0),
    ):
        if abs(low_offset) > 1e-12 or abs(high_offset) > 1e-12:
            output.append(
                {
                    "kind": "piecewise_offset",
                    "breakpoint": breakpoint,
                    "low_offset": float(low_offset),
                    "high_offset": float(high_offset),
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = _rooted(args.primary_cache)
    residual_path = _rooted(args.residual_cache)
    baselines, truth_series, index, _ = load_frozen_validation_baselines(
        primary_path,
        residual_path,
        _rooted(args.group3_cache),
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
        anchor = {target: active[target].copy() for target in TARGETS}
        anchor["kpx_group_1"] = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=args.group1_weight,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        anchor[TARGET] = apply_bounded_blend(
            primary[f"{TARGET}__reference"],
            primary[f"{TARGET}__expert"],
            weight=args.group2_weight,
            capacity=CAPACITY,
        )

    periods = period_rows(index)
    months = interval_month(index)
    records: list[dict[str, Any]] = []
    candidate_values: dict[int, np.ndarray] = {}
    for policy in _policies():
        calibrated = apply_policy(anchor[TARGET], policy)
        candidate = {target: anchor[target].copy() for target in TARGETS}
        candidate[TARGET] = calibrated
        q1 = metric_delta(truth, anchor, candidate, periods["q1"])
        q1_months = {
            month: metric_delta(
                truth,
                anchor,
                candidate,
                periods["q1"] & (months == month),
            )["score"]
            for month in (1, 2, 3)
        }
        q1_eligible = bool(
            all(q1[component] > 0.0 for component in COMPONENTS)
            and all(value >= 0.0 for value in q1_months.values())
        )
        q2 = None
        q2_eligible = False
        if q1_eligible:
            q2 = metric_delta(truth, anchor, candidate, periods["q2"])
            q2_months = {
                month: metric_delta(
                    truth,
                    anchor,
                    candidate,
                    periods["q2"] & (months == month),
                )["score"]
                for month in (4, 5, 6)
            }
            q2_eligible = bool(
                all(q2[component] > 0.0 for component in COMPONENTS)
                and all(value >= 0.0 for value in q2_months.values())
            )
        record = {
            "policy": policy,
            "q1": q1,
            "q1_months": q1_months,
            "q1_eligible": q1_eligible,
            "q2_if_q1_eligible": q2,
            "q2_transfer_eligible": q2_eligible,
            "movement": movement_summary(anchor, candidate),
        }
        records.append(record)
        candidate_values[id(record)] = calibrated

    transferable = [record for record in records if record["q2_transfer_eligible"]]
    selected = (
        max(
            transferable,
            key=lambda record: (
                min(record["q1"]["score"], record["q2_if_q1_eligible"]["score"]),
                record["q1"]["score"] + record["q2_if_q1_eligible"]["score"],
                -record["movement"]["mean_ratio"],
            ),
        )
        if transferable
        else None
    )
    confirmation = None
    if selected is not None:
        candidate = {target: anchor[target].copy() for target in TARGETS}
        candidate[TARGET] = candidate_values[id(selected)]
        period_deltas = {
            name: metric_delta(truth, anchor, candidate, rows)
            for name, rows in periods.items()
        }
        monthly = {
            str(month): metric_delta(
                truth, anchor, candidate, months == month
            )
            for month in range(1, 13)
        }
        confirmation = {
            "period_deltas": period_deltas,
            "monthly_deltas": monthly,
            "positive_score_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "h2_all_components_positive": bool(
                all(
                    period_deltas["h2"][component] > 0.0
                    for component in COMPONENTS
                )
            ),
            "movement": movement_summary(anchor, candidate),
        }

    report = {
        "schema_version": "public_positive_threshold_calibration.v1",
        "contract": {
            "selection": "Q1",
            "transfer": "Q2",
            "confirmation": "locked H2",
            "target": TARGET,
            "group1_frozen": True,
            "group3_frozen": True,
            "submission_side_effect": False,
        },
        "configuration": {
            "group1_weight": args.group1_weight,
            "group2_weight": args.group2_weight,
        },
        "searched_policies": len(records),
        "q1_eligible_policies": int(sum(record["q1_eligible"] for record in records)),
        "q2_transferable_policies": len(transferable),
        "selected": selected,
        "confirmation": confirmation,
        "decision": (
            "research_pass_no_production_writer"
            if confirmation is not None
            and confirmation["h2_all_components_positive"]
            else "rejected_fail_closed"
        ),
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
    parser.add_argument("--group1-weight", type=float, default=0.1375)
    parser.add_argument("--group2-weight", type=float, default=0.1825)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_threshold_calibration_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "searched_policies": report["searched_policies"],
                "q1_eligible_policies": report["q1_eligible_policies"],
                "q2_transferable_policies": report[
                    "q2_transferable_policies"
                ],
                "selected": report["selected"],
                "confirmation": report["confirmation"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
