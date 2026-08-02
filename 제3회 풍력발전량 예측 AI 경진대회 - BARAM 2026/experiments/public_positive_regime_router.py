"""Screen deployable regimes for the public-positive group-2 expansion.

The scored 1508365 candidate is the anchor.  A stronger group-2 factor has a
large FiCR gain but a small NMAE loss.  This experiment searches only simple
rules based on the anchor forecast and factor displacement, selects on Q1,
checks transfer on Q2, and opens H2 only for the locked rule.  Group 1 and
group 3 remain fixed at submission 1508365.
"""

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


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def regime_mask(
    anchor_ratio: np.ndarray,
    movement_ratio: np.ndarray,
    *,
    power_lower: float,
    power_upper: float,
    absolute_lower: float,
    absolute_upper: float,
    direction: str,
) -> np.ndarray:
    """Return a deterministic mask available on the submission horizon."""
    anchor_ratio = np.asarray(anchor_ratio, dtype=float)
    movement_ratio = np.asarray(movement_ratio, dtype=float)
    if anchor_ratio.shape != movement_ratio.shape:
        raise ValueError("router vectors do not align")
    if not 0.0 <= power_lower < power_upper:
        raise ValueError("invalid power interval")
    if not 0.0 <= absolute_lower < absolute_upper:
        raise ValueError("invalid movement interval")
    if direction not in ("all", "up", "down"):
        raise ValueError("unknown movement direction")
    mask = (
        (anchor_ratio >= power_lower)
        & (anchor_ratio < power_upper)
        & (np.abs(movement_ratio) >= absolute_lower)
        & (np.abs(movement_ratio) < absolute_upper)
    )
    if direction == "up":
        mask &= movement_ratio > 0.0
    elif direction == "down":
        mask &= movement_ratio < 0.0
    return mask


def apply_router(
    anchor: dict[str, np.ndarray],
    aggressive_group2: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {target: anchor[target].copy() for target in TARGETS}
    output[TARGET][mask] = np.asarray(aggressive_group2, dtype=float)[mask]
    for target in ("kpx_group_1", "kpx_group_3"):
        if not np.array_equal(output[target], anchor[target]):
            raise AssertionError(f"non-target changed: {target}")
    return output


def _rules() -> list[dict[str, float | str]]:
    power_bounds = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01)
    power_intervals = [
        (lower, upper)
        for lower, upper in itertools.combinations(power_bounds, 2)
        if upper - lower >= 0.20
    ]
    absolute_bounds = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 1.0)
    absolute_intervals = [
        (lower, upper)
        for lower, upper in itertools.combinations(absolute_bounds, 2)
    ]
    return [
        {
            "power_lower": power_lower,
            "power_upper": power_upper,
            "absolute_lower": absolute_lower,
            "absolute_upper": absolute_upper,
            "direction": direction,
        }
        for (power_lower, power_upper), (absolute_lower, absolute_upper), direction
        in itertools.product(
            power_intervals,
            absolute_intervals,
            ("all", "up", "down"),
        )
    ]


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
            weight=args.anchor_group2_weight,
            capacity=CAPACITY_KWH[TARGET],
        )
        aggressive = apply_bounded_blend(
            primary[f"{TARGET}__reference"],
            primary[f"{TARGET}__expert"],
            weight=args.aggressive_group2_weight,
            capacity=CAPACITY_KWH[TARGET],
        )

    anchor_ratio = anchor[TARGET] / CAPACITY_KWH[TARGET]
    movement_ratio = (aggressive - anchor[TARGET]) / CAPACITY_KWH[TARGET]
    periods = period_rows(index)
    months = interval_month(index)
    records: list[dict[str, Any]] = []
    masks: dict[int, np.ndarray] = {}
    for position, rule in enumerate(_rules()):
        mask = regime_mask(
            anchor_ratio,
            movement_ratio,
            **rule,
        )
        coverage = float(mask.mean())
        if not args.minimum_coverage <= coverage <= args.maximum_coverage:
            continue
        candidate = apply_router(anchor, aggressive, mask)
        q1 = metric_delta(truth, anchor, candidate, periods["q1"])
        q1_months = {
            str(month): metric_delta(
                truth,
                anchor,
                candidate,
                periods["q1"] & (months == month),
            )
            for month in (1, 2, 3)
        }
        q1_eligible = bool(
            all(q1[component] > 0.0 for component in COMPONENTS)
            and all(value["score"] >= 0.0 for value in q1_months.values())
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
            "rule": rule,
            "coverage": coverage,
            "q1": q1,
            "q1_months": q1_months,
            "q1_eligible": q1_eligible,
            "q2_if_q1_eligible": q2,
            "q2_transfer_eligible": q2_eligible,
        }
        records.append(record)
        masks[id(record)] = mask

    transferable = [record for record in records if record["q2_transfer_eligible"]]
    selected = (
        max(
            transferable,
            key=lambda record: (
                min(record["q1"]["score"], record["q2_if_q1_eligible"]["score"]),
                record["q1"]["score"] + record["q2_if_q1_eligible"]["score"],
                -record["coverage"],
            ),
        )
        if transferable
        else None
    )
    confirmation = None
    if selected is not None:
        mask = masks[id(selected)]
        candidate = apply_router(anchor, aggressive, mask)
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
            "movement": movement_summary(anchor, candidate),
            "h2_all_components_positive": bool(
                all(
                    period_deltas["h2"][component] > 0.0
                    for component in COMPONENTS
                )
            ),
        }

    report = {
        "schema_version": "public_positive_regime_router.v1",
        "contract": {
            "selection": "Q1",
            "transfer": "Q2",
            "confirmation": "locked H2",
            "anchor_submission_id": 1508365,
            "group1_frozen": True,
            "group3_frozen": True,
            "submission_side_effect": False,
        },
        "configuration": {
            "group1_weight": args.group1_weight,
            "anchor_group2_weight": args.anchor_group2_weight,
            "aggressive_group2_weight": args.aggressive_group2_weight,
            "minimum_coverage": args.minimum_coverage,
            "maximum_coverage": args.maximum_coverage,
        },
        "searched_rules": len(records),
        "q1_eligible_rules": int(sum(record["q1_eligible"] for record in records)),
        "q2_transferable_rules": len(transferable),
        "selected": selected,
        "confirmation": confirmation,
        "decision": (
            "research_pass_no_production_writer"
            if confirmation is not None
            and confirmation["h2_all_components_positive"]
            else "rejected_fail_closed"
        ),
        "top_transferable": sorted(
            transferable,
            key=lambda record: min(
                record["q1"]["score"],
                record["q2_if_q1_eligible"]["score"],
            ),
            reverse=True,
        )[:20],
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
    parser.add_argument("--anchor-group2-weight", type=float, default=0.095)
    parser.add_argument("--aggressive-group2-weight", type=float, default=0.1825)
    parser.add_argument("--minimum-coverage", type=float, default=0.05)
    parser.add_argument("--maximum-coverage", type=float, default=0.90)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_regime_router_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "searched_rules": report["searched_rules"],
                "q1_eligible_rules": report["q1_eligible_rules"],
                "q2_transferable_rules": report["q2_transferable_rules"],
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
