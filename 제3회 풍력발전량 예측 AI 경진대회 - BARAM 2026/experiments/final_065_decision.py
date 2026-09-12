"""Fail-closed final screening of the remaining paths to a 0.65 public score.

This module does not write a submission.  It applies the corrected central
promotion policy to the only two still-materialized, unsubmitted follow-ups
and quantifies the remaining gap from the frozen public incumbent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_service.contracts import Evaluation
from agent_service.policy import PromotionPolicy


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(_rooted(path).read_text(encoding="utf-8"))


def conservative_subset_q05(
    report: dict[str, Any],
) -> dict[str, float]:
    """Use the weaker q05 across IID and month-stratified complements."""
    output: dict[str, float] = {}
    schemes = (
        report["iid_timestamp_splits"],
        report["month_stratified_timestamp_splits"],
    )
    for split in ("public", "private"):
        for component in COMPONENTS:
            output[f"subset_{split}_{component}_q05"] = float(
                min(scheme[split][component]["q05"] for scheme in schemes)
            )
    return output


def _movement_ratio(candidate: dict[str, Any]) -> float:
    changed = candidate["candidate"]["composition"]["changed_rows"]
    if isinstance(changed, dict):
        return float(max(changed.values()) / 8_760)
    return float(changed / 8_760)


def group12_evaluation(
    candidate: dict[str, Any],
    subset: dict[str, Any],
) -> Evaluation:
    validation = candidate["validation"]
    full = validation["period_deltas"]["full"]
    h2 = validation["period_deltas"]["h2"]
    months = validation["monthly_deltas"]
    bootstrap = validation["issue_block_bootstrap"]
    monthly_scores = [float(row["score"]) for row in months.values()]
    return Evaluation.from_dict(
        {
            "family": "group12_difference_reconciliation",
            "family_group": "broad_group12_postprocessing",
            "direction": "zero_sum_reconciliation",
            "locked_score_delta": float(h2["score"]),
            "locked_one_minus_nmae_delta": float(h2["one_minus_nmae"]),
            "locked_ficr_delta": float(h2["ficr"]),
            # The report is the affected-pair mean; two of three macro groups move.
            "expected_macro_score_delta": float(full["score"]) * 2.0 / 3.0,
            "positive_months": int(np.sum(np.asarray(monthly_scores) > 0.0)),
            "total_months": int(len(monthly_scores)),
            "worst_month_score_delta": float(min(monthly_scores)),
            "bootstrap_positive_fraction": float(
                bootstrap["positive_fraction"]
            ),
            "bootstrap_q05": float(bootstrap["q05"]),
            "changed_ratio": _movement_ratio(candidate),
            "p95_movement_ratio": float(
                candidate["orthogonality"][
                    "maximum_absolute_movement_capacity_ratio"
                ]
            ),
            "leakage_risk": "medium",
            "rule_violation": "none",
            **conservative_subset_q05(subset),
        }
    )


def group2_expansion_evaluation(candidate: dict[str, Any]) -> Evaluation:
    validation = candidate["validation"]
    issue = validation["issue_block_validation_incremental"]
    h2 = validation["period_incremental_deltas_vs_incumbent_factor"]["h2"]
    monthly_scores = [
        float(row["score"])
        for row in validation["monthly_incremental_deltas"].values()
    ]
    bootstrap = issue["issue_block_bootstrap"]
    movement = candidate["movement_vs_incumbent"]
    return Evaluation.from_dict(
        {
            "family": "public_positive_pooled_group2_factor_expansion",
            "family_group": "pooled_group2_expansion",
            "direction": "same_direction_expansion",
            "locked_score_delta": float(h2["score"]),
            "locked_one_minus_nmae_delta": float(h2["one_minus_nmae"]),
            "locked_ficr_delta": float(h2["ficr"]),
            "expected_macro_score_delta": float(
                candidate["public_transfer_projection"][
                    "projected_incremental_macro_delta"
                ]
            ),
            "positive_months": int(np.sum(np.asarray(monthly_scores) > 0.0)),
            "total_months": int(len(monthly_scores)),
            "worst_month_score_delta": float(min(monthly_scores)),
            "bootstrap_positive_fraction": float(
                bootstrap["positive_fraction"]
            ),
            "bootstrap_q05": float(bootstrap["q05"]),
            "changed_ratio": float(movement["changed_rows"] / 8_760),
            "p95_movement_ratio": float(movement["p95_capacity_ratio"]),
            "leakage_risk": "medium",
            "rule_violation": "none",
            # Deliberately no subset q05 fields: missing evidence must fail closed.
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _read(args.config)
    policy = PromotionPolicy(
        config["policy"],
        human_submission_required=True,
    )
    audit = _read(args.evaluation_audit)
    factors = _read(args.factor_audit)
    subset = _read(args.subset_stress)
    group12 = _read(args.group12_candidate)
    group2 = _read(args.group2_candidate)

    incumbent = audit["leaderboard"]["best"]
    target_score = float(args.target_score)
    gap = target_score - float(incumbent["score"])
    retained = factors["summary"]["retained_public_positive_axes"]
    retained_gain = float(
        sum(factors["factors"][name]["macro_delta"]["score"] for name in retained)
    )
    decisions = {
        "group12_difference_reconciliation": policy.evaluate(
            group12_evaluation(group12, subset)
        ).to_dict(),
        "pooled_group2_weight_010_expansion": policy.evaluate(
            group2_expansion_evaluation(group2)
        ).to_dict(),
    }
    projected_group2 = float(
        group2["public_transfer_projection"]["projected_public_score"]
    )
    report = {
        "family": "final_065_fail_closed_decision",
        "target_score": target_score,
        "incumbent": incumbent,
        "gap_to_target": gap,
        "metric_audit": {
            "passed": bool(
                audit["official_metric"]["exact_within_1e_12"]
                and audit["senior_assessment"]["metric_implementation"]
                == "passed"
            ),
            "diagnosis": audit["senior_assessment"]["primary_bottleneck"],
        },
        "corrected_policy": {
            "version": policy.version,
            "requires_40_60_subset_stress": bool(
                config["policy"]["require_public_private_subset_stress"]
            ),
            "minimum_subset_q05": float(
                config["policy"]["min_public_private_subset_q05"]
            ),
            "minimum_issue_bootstrap_q05": float(
                config["policy"]["min_bootstrap_q05"]
            ),
        },
        "remaining_materialized_candidates": decisions,
        "score_scale": {
            "retained_public_factor_gain_sum": retained_gain,
            "target_gap_over_retained_gain_sum": (
                gap / retained_gain if retained_gain > 0.0 else None
            ),
            "best_pending_projection": projected_group2,
            "best_pending_projection_gap": target_score - projected_group2,
            "interpretation": (
                "The confirmed factor axes are micro-gains already present in "
                "the incumbent; repeated scaling is not a validated path to 0.65."
            ),
        },
        "decision": {
            "submission_candidate_created": False,
            "all_materialized_candidates_rejected": bool(
                all(item["outcome"] == "rejected" for item in decisions.values())
            ),
            "incumbent_frozen": True,
            "incumbent_submission_id": incumbent["submission_id"],
            "next_admissible_family": (
                "a genuinely new core model with causal 2023 and 2024 forward "
                "OOF plus non-negative component q05 on exact 40/60 complements"
            ),
            "closed_families": factors["summary"]["frozen_axes"],
        },
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
    parser.add_argument("--config", default=".agents/baram.json")
    parser.add_argument(
        "--evaluation-audit",
        default=(
            "artifacts_final/diagnostics/"
            "evaluation_structure_audit_20260729.json"
        ),
    )
    parser.add_argument(
        "--factor-audit",
        default=(
            "artifacts_final/diagnostics/"
            "public_factor_response_audit_20260728.json"
        ),
    )
    parser.add_argument(
        "--subset-stress",
        default=(
            "artifacts_final/diagnostics/"
            "public_private_subset_stress_20260729.json"
        ),
    )
    parser.add_argument(
        "--group12-candidate",
        default=(
            "artifacts_final/diagnostics/"
            "group12_difference_reconciliation_unanimous_w10_20260728.json"
        ),
    )
    parser.add_argument(
        "--group2-candidate",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_pooled_g2_w10_probe_20260727.json"
        ),
    )
    parser.add_argument("--target-score", type=float, default=0.65)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "final_065_decision_20260729.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "target_score": report["target_score"],
                "incumbent": report["incumbent"],
                "gap_to_target": report["gap_to_target"],
                "remaining_materialized_candidates": (
                    report["remaining_materialized_candidates"]
                ),
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
