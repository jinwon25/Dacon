from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_service.config import ServiceConfig
from agent_service.contracts import (
    Evaluation,
    Hypothesis,
    RunSpec,
    ValidationPlan,
    resolve_inside,
)
from agent_service.policy import PolicyDecision, PromotionPolicy
from agent_service.runner import RunResult, SafeModuleRunner
from agent_service.store import AgentStore


class Orchestrator:
    def __init__(self, config: ServiceConfig, store: AgentStore):
        self.config = config
        self.store = store
        self.policy = PromotionPolicy(
            config.policy,
            human_submission_required=config.human_submission_required,
        )
        self.runner = SafeModuleRunner(config)

    def initialize(self) -> None:
        self.store.initialize()
        if "competition_profile_path" in self.config.raw:
            self.store.register_competition(self.config.competition_profile)
        self.config.run_root.mkdir(parents=True, exist_ok=True)
        self.config.submission_dir.mkdir(parents=True, exist_ok=True)
        self.config.archive_dir.mkdir(parents=True, exist_ok=True)

    def propose(self, hypothesis: Hypothesis) -> int:
        if hypothesis.competition_slug != self.config.competition_slug:
            raise ValueError(
                f"hypothesis competition {hypothesis.competition_slug} does not match "
                f"service competition {self.config.competition_slug}"
            )
        hypothesis_id = self.store.create_hypothesis(hypothesis)
        self.store.create_task(
            "modeling",
            {
                "hypothesis_id": hypothesis_id,
                "contract": self.config.roles["modeling"],
                "hypothesis": hypothesis.to_dict(),
            },
        )
        return hypothesis_id

    def register_run(self, spec: RunSpec) -> int:
        self.runner.validate(spec)
        if self.config.governance.get("require_validation_approval", False):
            if spec.validation_plan_id is None:
                raise ValueError("an approved validation_plan_id is required")
            if not self.store.is_approved(
                "validation_strategy", "validation_plan", spec.validation_plan_id
            ):
                raise ValueError("the validation strategy has not been approved")
        return self.store.create_run(spec)

    def register_validation_plan(self, plan: ValidationPlan) -> int:
        if plan.competition_slug != self.config.competition_slug:
            raise ValueError(
                f"validation plan competition {plan.competition_slug} does not match "
                f"service competition {self.config.competition_slug}"
            )
        return self.store.create_validation_plan(plan)

    def approve(
        self,
        gate_type: str,
        subject_type: str,
        subject_id: int,
        decision: str,
        reviewer: str,
        reason: str,
    ) -> int:
        return self.store.record_approval(
            gate_type, subject_type, subject_id, decision, reviewer, reason
        )

    def select_run(
        self,
        run_id: int,
        selection_type: str,
        rationale: str,
        selected_by: str,
    ) -> int:
        if selection_type == "submission_candidate":
            decision = self.store.get_latest_decision(run_id)
            if decision is None or decision.get("outcome") != "candidate":
                raise ValueError(
                    "a submission candidate must pass deterministic promotion first"
                )
        return self.store.select_run(
            self.config.competition_slug,
            selection_type,
            run_id,
            rationale,
            selected_by,
        )

    def _update_local_best(self, run_id: int, evaluation: Evaluation) -> None:
        current = self.store.get_active_selection(
            self.config.competition_slug, "local_best"
        )
        candidate_metric = (
            evaluation.selection_metric
            if evaluation.selection_metric is not None
            else evaluation.expected_macro_score_delta
        )
        better = current is None
        if current is not None:
            previous_payload = self.store.get_evaluation(int(current["run_id"]))
            if previous_payload is None:
                better = True
            else:
                previous = Evaluation.from_dict(previous_payload)
                previous_metric = (
                    previous.selection_metric
                    if previous.selection_metric is not None
                    else previous.expected_macro_score_delta
                )
                better = (
                    candidate_metric > previous_metric
                    if evaluation.selection_direction == "maximize"
                    else candidate_metric < previous_metric
                )
        if better:
            self.select_run(
                run_id,
                "local_best",
                f"deterministic local metric={candidate_metric:.12g}",
                "promotion_policy",
            )

    def execute(self, run_id: int) -> RunResult:
        run = self.store.get_run(run_id)
        if run["status"] not in {"pending", "failed", "timed_out"}:
            raise ValueError(f"Run {run_id} cannot execute from {run['status']}")
        spec = RunSpec.from_dict(run["spec"])
        materialized = spec.materialize(run_id)
        self.store.set_run_status(run_id, "running")
        attempt_no = self.store.start_run_attempt(run_id)
        result = self.runner.run(run_id, spec, attempt_no=attempt_no)
        attempt_status = (
            "completed"
            if result.exit_code == 0 and result.expected_outputs_present
            else ("timed_out" if result.timed_out else "failed")
        )
        self.store.finish_run_attempt(
            run_id, attempt_no, attempt_status, result.manifest_path
        )
        if result.exit_code == 0 and result.expected_outputs_present:
            self.store.set_run_status(run_id, "validation_pending", result.exit_code)
            self.store.create_task(
                "validation",
                {
                    "run_id": run_id,
                    "contract": self.config.roles["validation"],
                    "evaluation_path": materialized.evaluation_path,
                    "report_path": materialized.report_path,
                },
                run_id=run_id,
            )
        else:
            status = "timed_out" if result.timed_out else "failed"
            self.store.set_run_status(run_id, status, result.exit_code)
        return result

    def evaluate(self, run_id: int, evaluation: Evaluation | None = None) -> PolicyDecision:
        run = self.store.get_run(run_id)
        if run["status"] not in {"validation_pending", "pending", "imported"}:
            raise ValueError(f"Run {run_id} cannot validate from {run['status']}")
        spec = RunSpec.from_dict(run["spec"])
        spec = spec.materialize(run_id)
        if evaluation is None:
            path = resolve_inside(
                self.config.project_root, spec.evaluation_path, "evaluation_path"
            )
            evaluation = Evaluation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        uses_external_data = bool(spec.tags.get("external_data"))
        if uses_external_data:
            compliance_errors: list[str] = []
            if not spec.external_manifest_paths:
                compliance_errors.append("external data manifest is missing")
            else:
                from agent_service.compliance import validate_external_data_manifest

                for manifest_path in spec.external_manifest_paths:
                    try:
                        validate_external_data_manifest(
                            Path(manifest_path), self.config.project_root
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        compliance_errors.append(str(exc))
            if compliance_errors:
                evaluation = replace(
                    evaluation,
                    leakage_risk="high",
                    rule_violation="external_data_provenance_failed: "
                    + " | ".join(compliance_errors),
                )
        failed_coverage = self.store.latest_failed_family_coverage(evaluation.family)
        failed_evidence = self.store.latest_failed_family_evidence(
            evaluation.family_group or evaluation.family,
            evaluation.direction,
        )
        decision = self.policy.evaluate(
            evaluation,
            failed_coverage,
            public_failure_evidence=failed_evidence,
        )
        self.store.record_evaluation(run_id, evaluation)
        self.store.record_decision(run_id, decision)
        if decision.outcome == "candidate":
            self._update_local_best(run_id, evaluation)
            status = (
                "candidate_pending_human"
                if decision.human_submission_required
                else "auto_submit_ready"
            )
            if not decision.human_submission_required:
                self.select_run(
                    run_id,
                    "submission_candidate",
                    "passed deterministic promotion and safety gates",
                    "promotion_policy",
                )
        else:
            status = "rejected"
        self.store.set_run_status(run_id, status)
        self.store.create_task(
            "steward",
            {
                "run_id": run_id,
                "contract": self.config.roles["steward"],
                "decision": decision.to_dict(),
                "candidate_path": spec.candidate_path,
            },
            run_id=run_id,
        )
        return decision

    def archive_rejected(self, run_id: int, apply: bool = False) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run["status"] not in {"rejected", "public_rejected", "archived"}:
            raise ValueError(f"Run {run_id} is not rejected")
        spec = RunSpec.from_dict(run["spec"])
        spec = spec.materialize(run_id)
        if not spec.candidate_path:
            return {"run_id": run_id, "action": "none", "reason": "no candidate_path"}
        source = resolve_inside(
            self.config.project_root, spec.candidate_path, "candidate_path"
        )
        try:
            source.relative_to(self.config.submission_dir)
        except ValueError as exc:
            raise ValueError("Rejected candidate must be inside submissions/") from exc
        if source.parent == self.config.archive_dir:
            return {"run_id": run_id, "action": "none", "path": str(source)}
        destination = self.config.archive_dir / source.name
        if destination.exists() and source.exists():
            destination = self.config.archive_dir / f"{source.stem}_run{run_id}{source.suffix}"
        result = {
            "run_id": run_id,
            "action": "archive",
            "source": str(source.relative_to(self.config.project_root)),
            "destination": str(destination.relative_to(self.config.project_root)),
            "applied": apply,
        }
        if apply and source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            self.store.set_run_status(run_id, "archived")
        return result

    def cleanup_submission_artifacts(self, apply: bool = False) -> dict[str, Any]:
        """Inventory submissions and archive byte-identical duplicates.

        The active candidate and the newest copy of each hash are retained.  This
        is deliberately conservative: only files inside ``submissions/`` are
        considered and no artifact is deleted, merely moved to ``archive/``.
        """
        files = sorted(self.config.submission_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        groups: dict[str, list[Path]] = {}
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path)
        duplicates: list[dict[str, str]] = []
        for digest, paths in groups.items():
            if len(paths) < 2:
                continue
            # newest is retained; older copies are recoverably archived
            for source in paths[:-1]:
                destination = self.config.archive_dir / source.name
                if destination.exists():
                    destination = self.config.archive_dir / f"{source.stem}_{digest[:8]}{source.suffix}"
                item = {"source": str(source.relative_to(self.config.project_root)),
                        "destination": str(destination.relative_to(self.config.project_root)),
                        "sha256": digest, "applied": bool(apply)}
                if apply:
                    self.config.archive_dir.mkdir(parents=True, exist_ok=True)
                    source.replace(destination)
                duplicates.append(item)
        return {"scanned": len(files), "duplicate_count": len(duplicates), "duplicates": duplicates, "applied": bool(apply)}

    def record_public_result(self, payload: dict[str, Any]) -> None:
        self.store.record_public_result(payload)
        run_id = payload.get("run_id")
        if run_id is not None:
            run_id = int(run_id)
            status = (
                "public_selected"
                if float(payload["score_delta_vs_best"]) >= 0.0
                else "public_rejected"
            )
            self.store.set_run_status(run_id, status)
            # A scored file must never remain eligible for another automatic
            # submission. A public loss also invalidates its local-best status;
            # otherwise the planner keeps branching from evidence the public
            # leaderboard has already falsified.
            self.store.deactivate_selection(
                self.config.competition_slug,
                "submission_candidate",
                "public result was recorded",
                run_id=run_id,
            )
            if status == "public_rejected":
                self.store.deactivate_selection(
                    self.config.competition_slug,
                    "local_best",
                    "public score rejected the local selection",
                    run_id=run_id,
                )

    def submission_check(self, run_id: int, execute: bool = False) -> dict[str, Any]:
        from agent_service.submission import AutomaticSubmissionController

        return AutomaticSubmissionController(self.config, self.store).submit(
            run_id, execute=execute
        )

    def sync_leaderboard(self) -> dict[str, Any]:
        from agent_service.leaderboard import LeaderboardSynchronizer

        return LeaderboardSynchronizer(self.config, self).sync()

    def auto_cycle(self, execute_submissions: bool = False) -> dict[str, Any]:
        leaderboard = self.sync_leaderboard()
        submissions = []
        for run in reversed(self.store.list_rows("runs", limit=10_000)):
            if run["status"] in {"candidate_pending_human", "auto_submit_ready"}:
                submissions.append(
                    self.submission_check(int(run["id"]), execute=execute_submissions)
                )
        return {"leaderboard": leaderboard, "submissions": submissions}

    def status(self) -> dict[str, Any]:
        runs = self.store.list_rows("runs", limit=100_000)
        tasks = self.store.list_rows("tasks", limit=100_000)
        public_results = self.store.list_rows("public_results", limit=100_000)
        selections = [
            row
            for row in self.store.list_rows("selections", limit=100_000)
            if int(row["active"]) == 1
        ]
        run_counts: dict[str, int] = {}
        for run in runs:
            run_counts[run["status"]] = run_counts.get(run["status"], 0) + 1
        task_counts: dict[str, int] = {}
        for task in tasks:
            key = f"{task['role']}:{task['status']}"
            task_counts[key] = task_counts.get(key, 0) + 1
        public_best = (
            None
            if not public_results
            else max(public_results, key=lambda row: float(row["score"]))
        )
        return {
            "competition": (
                self.config.competition_profile.to_dict()
                if "competition_profile_path" in self.config.raw
                else {"slug": self.config.competition_slug}
            ),
            "run_counts": run_counts,
            "task_counts": task_counts,
            "active_selections": selections,
            "public_best": public_best,
            "auto_submission": {
                "enabled": bool(
                    self.config.submission.get("auto_submit_enabled", False)
                ),
                "human_submission_required": self.config.human_submission_required,
                "execute_requires_local_cli": True,
            },
        }
