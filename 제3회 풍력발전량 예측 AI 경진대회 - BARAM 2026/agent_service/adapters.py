from __future__ import annotations

from typing import Any

from agent_service.contracts import Evaluation
from src.metrics import CAPACITY_KWH


GROUP_3_CAPACITY = CAPACITY_KWH["kpx_group_3"]


def _worst_month_score(rows: dict[str, Any]) -> float | None:
    scores: list[float] = []
    for row in rows.values():
        values = row.get("delta", row) if isinstance(row, dict) else {}
        if "score" in values:
            scores.append(float(values["score"]))
    return min(scores) if scores else None


def phase_regime_evaluation(report: dict[str, Any]) -> Evaluation:
    locked = report["locked_validation"]
    delta = locked["metrics"]["delta"]
    bootstrap = locked["day_bootstrap"]
    final = report["final"]
    return Evaluation.from_dict(
        {
            "family": "phase_regime_cross_group",
            "locked_score_delta": delta["score"],
            "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
            "locked_ficr_delta": delta["ficr"],
            "expected_macro_score_delta": locked[
                "expected_competition_macro_score_delta"
            ],
            "positive_months": locked["positive_months"],
            "total_months": len(locked["monthly_deltas"]),
            "worst_month_score_delta": _worst_month_score(
                locked["monthly_deltas"]
            ),
            "bootstrap_positive_fraction": bootstrap["positive_fraction"],
            "bootstrap_q05": bootstrap["q05"],
            "changed_ratio": final["changed_ratio"],
            "p95_movement_ratio": final["p95_absolute_movement_kwh"]
            / GROUP_3_CAPACITY,
            "notes": "Imported from the phase/regime exact-OOF report.",
        }
    )


def meta_gate_evaluation(report: dict[str, Any]) -> Evaluation:
    locked = report["validation"]["locked_h1_to_h2"]["metrics"]
    delta = locked["delta"]
    bootstrap = report["locked_h2_day_bootstrap"]
    monthly = report["locked_h2_monthly_deltas"]
    final = report["final"]
    return Evaluation.from_dict(
        {
            "family": "exact_oof_meta_gate",
            "locked_score_delta": delta["score"],
            "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
            "locked_ficr_delta": delta["ficr"],
            "expected_macro_score_delta": delta["score"] / 3.0,
            "positive_months": sum(row["score"] > 0.0 for row in monthly.values()),
            "total_months": len(monthly),
            "worst_month_score_delta": _worst_month_score(monthly),
            "bootstrap_positive_fraction": bootstrap["positive_fraction"],
            "bootstrap_q05": bootstrap["q05"],
            "changed_ratio": final["changed_ratio"],
            "p95_movement_ratio": final["p95_absolute_movement_kwh"]
            / GROUP_3_CAPACITY,
            "notes": "Imported from the exact-OOF meta-gate report.",
        }
    )


def meta_gate_sweep_evaluation(report: dict[str, Any]) -> Evaluation:
    """Adapt a fine-sweep candidate relative to the incumbent meta-gate.

    The sweep report contains absolute deltas versus the pre-gate trajectory as
    diagnostics. Promotion must instead use the paired incremental result versus
    the already-public p55/a25 incumbent; otherwise the service double-counts the
    incumbent gain.
    """
    incremental = report["locked_incremental_over_reference"]
    delta = incremental["metrics"]["delta"]
    bootstrap = incremental["day_bootstrap"]
    monthly = incremental["monthly_deltas"]
    final = report.get("submission")
    if not isinstance(final, dict):
        raise ValueError("Meta-gate sweep report has no generated submission candidate")
    selected = incremental["selected_policy"]
    reference = incremental["reference_policy"]
    return Evaluation.from_dict(
        {
            "family": "exact_oof_meta_gate_sweep",
            "locked_score_delta": delta["score"],
            "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
            "locked_ficr_delta": delta["ficr"],
            "expected_macro_score_delta": delta["score"] / 3.0,
            "positive_months": sum(row["score"] > 0.0 for row in monthly.values()),
            "total_months": len(monthly),
            "worst_month_score_delta": _worst_month_score(monthly),
            "bootstrap_positive_fraction": bootstrap["positive_fraction"],
            "bootstrap_q05": bootstrap["q05"],
            "changed_ratio": final["changed_ratio"],
            "p95_movement_ratio": final["p95_absolute_movement_kwh"]
            / GROUP_3_CAPACITY,
            "leakage_risk": "low",
            "rule_violation": "none",
            "notes": (
                "Imported from the leakage-safe exact-OOF meta-gate fine sweep; "
                f"selected p={selected['threshold']}, a={selected['alpha']} versus "
                f"reference p={reference['threshold']}, a={reference['alpha']}; "
                f"candidate_sha256={final.get('file_sha256', 'not-recorded')}."
            ),
        }
    )


def spatiotemporal_multitask_evaluation(report: dict[str, Any]) -> Evaluation:
    """Adapt the locked deployable ensemble, not a seed-level diagnostic."""
    locked = report["locked_h2_ensemble"]
    delta = locked["overall"]["delta"]
    bootstrap = locked["issue_block_bootstrap"]
    monthly = locked["monthly"]
    final = report.get("submission")
    if not isinstance(final, dict):
        raise ValueError("Spatiotemporal promotion report has no generated candidate")
    audit = final.get("candidate_validator", {})
    if not audit.get("valid", False):
        raise ValueError("Spatiotemporal candidate did not pass CandidateValidator")
    return Evaluation.from_dict(
        {
            "family": "spatiotemporal_multitask_blend",
            "locked_score_delta": delta["score"],
            "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
            "locked_ficr_delta": delta["ficr"],
            "expected_macro_score_delta": delta["score"] / 3.0,
            "positive_months": sum(
                row["delta"]["score"] > 0.0 for row in monthly.values()
            ),
            "total_months": len(monthly),
            "worst_month_score_delta": _worst_month_score(monthly),
            "bootstrap_positive_fraction": bootstrap["positive_fraction"],
            "bootstrap_q05": bootstrap["q05"],
            "changed_ratio": final["changed_ratio"],
            "p95_movement_ratio": final["p95_absolute_movement_kwh"]
            / GROUP_3_CAPACITY,
            "leakage_risk": "low",
            "rule_violation": "none",
            "notes": (
                "Imported from the Q1/Q2-selected, locked-H2 spatiotemporal ensemble; "
                f"candidate_sha256={final['sha256']}."
            ),
        }
    )


ADAPTERS = {
    "phase_regime": phase_regime_evaluation,
    "meta_gate": meta_gate_evaluation,
    "meta_gate_sweep": meta_gate_sweep_evaluation,
    "spatiotemporal_multitask": spatiotemporal_multitask_evaluation,
}


def adapt_report(adapter: str, report: dict[str, Any]) -> Evaluation:
    try:
        function = ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown report adapter: {adapter}") from exc
    return function(report)
