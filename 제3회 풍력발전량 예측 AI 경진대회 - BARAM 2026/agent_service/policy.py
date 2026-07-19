from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_service.contracts import Evaluation


@dataclass(frozen=True)
class PolicyDecision:
    outcome: str
    reasons: tuple[str, ...]
    policy_version: str
    human_submission_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "policy_version": self.policy_version,
            "human_submission_required": self.human_submission_required,
        }


class PromotionPolicy:
    def __init__(self, settings: dict[str, Any], human_submission_required: bool = True):
        self.settings = settings
        self.version = str(settings.get("version", "unversioned"))
        self.human_submission_required = human_submission_required

    def _threshold(self, name: str, family: str) -> float:
        """Return an explicitly versioned family threshold when configured.

        Sparse post-processing corrections and structural model blends have
        different, measurable movement footprints.  Family overrides let the
        service express that distinction without weakening the score,
        component, temporal, or bootstrap gates for every experiment.
        """
        overrides = self.settings.get("family_overrides", {})
        family_settings = overrides.get(family, {}) if isinstance(overrides, dict) else {}
        value = family_settings.get(name, self.settings[name])
        return float(value)

    def evaluate(
        self,
        evaluation: Evaluation,
        latest_failed_family_coverage: float | None = None,
        public_failure_evidence: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        checks = (
            (
                evaluation.leakage_risk != "high",
                "leakage risk is high",
            ),
            (
                evaluation.rule_violation in {"", "none"},
                "competition rule violation is present",
            ),
            (
                evaluation.locked_score_delta
                >= self._threshold("min_locked_score_delta", evaluation.family),
                "locked score delta is below the minimum",
            ),
            (
                evaluation.locked_one_minus_nmae_delta
                >= self._threshold("min_locked_one_minus_nmae_delta", evaluation.family),
                "locked 1-NMAE delta is negative",
            ),
            (
                evaluation.locked_ficr_delta
                >= self._threshold("min_locked_ficr_delta", evaluation.family),
                "locked FICR delta is negative",
            ),
            (
                evaluation.expected_macro_score_delta
                >= self._threshold("min_expected_macro_score_delta", evaluation.family),
                "expected macro score delta is too small",
            ),
            (
                evaluation.positive_month_fraction
                >= self._threshold("min_positive_month_fraction", evaluation.family),
                "too few locked months improved",
            ),
            (
                evaluation.bootstrap_positive_fraction
                >= self._threshold("min_bootstrap_positive_fraction", evaluation.family),
                "day-bootstrap positive fraction is too low",
            ),
            (
                evaluation.bootstrap_q05
                >= self._threshold("min_bootstrap_q05", evaluation.family),
                "day-bootstrap lower tail is too negative",
            ),
            (
                evaluation.changed_ratio
                <= self._threshold("max_changed_ratio", evaluation.family),
                "candidate changes too many rows",
            ),
            (
                evaluation.p95_movement_ratio
                <= self._threshold("max_p95_movement_ratio", evaluation.family),
                "candidate movement is too large",
            ),
        )
        reasons = [message for passed, message in checks if not passed]
        require_worst_month = bool(
            self.settings.get("require_worst_month_score_delta", False)
        )
        minimum_worst_month_raw = self.settings.get("min_worst_month_score_delta")
        if evaluation.worst_month_score_delta is None:
            if require_worst_month:
                reasons.append("worst-month score evidence is missing")
        elif (
            minimum_worst_month_raw is not None
            and evaluation.worst_month_score_delta < float(minimum_worst_month_raw)
        ):
            reasons.append("worst-month score delta is negative")
        # Keep the original concrete-family guard for old callers and old DB
        # rows.  Newer callers can pass richer public evidence, allowing a
        # variant in the same method family/group and direction to be gated
        # without naming any submission file.
        evidence_coverage = latest_failed_family_coverage
        evidence_reason = "a publicly failed family must reduce row coverage by at least 75%"
        if public_failure_evidence is not None:
            guard = self.settings.get("public_failure_guard", {})
            enabled = bool(guard.get("enabled", True)) if isinstance(guard, dict) else True
            if enabled:
                candidate_group = evaluation.family_group or evaluation.family
                failed_group = str(
                    public_failure_evidence.get("family_group", "")
                ).strip()
                failed_direction = str(
                    public_failure_evidence.get("direction", "unknown")
                ).strip().lower() or "unknown"
                same_group = candidate_group == failed_group
                require_direction = bool(
                    guard.get("require_same_direction", True)
                ) if isinstance(guard, dict) else True
                same_direction = (
                    not require_direction
                    or (
                        evaluation.direction != "unknown"
                        and failed_direction != "unknown"
                        and evaluation.direction == failed_direction
                    )
                )
                if same_group and same_direction:
                    evidence_coverage = float(
                        public_failure_evidence["changed_ratio"]
                    )
                    evidence_reason = (
                        "a public failure in this method family/direction requires "
                        "a materially smaller row-coverage probe"
                    )
        if evidence_coverage is not None:
            guard = self.settings.get("public_failure_guard", {})
            fraction = (
                guard.get("max_coverage_fraction")
                if isinstance(guard, dict)
                else None
            )
            if fraction is None:
                fraction = self.settings["failed_family_max_coverage_fraction"]
            maximum = (
                evidence_coverage * float(fraction)
            )
            if evaluation.changed_ratio > maximum:
                reasons.append(evidence_reason)
        outcome = "candidate" if not reasons else "rejected"
        if outcome == "candidate" and self.human_submission_required:
            reasons.append("external submission requires human approval")
        return PolicyDecision(
            outcome=outcome,
            reasons=tuple(reasons),
            policy_version=self.version,
            human_submission_required=self.human_submission_required,
        )
