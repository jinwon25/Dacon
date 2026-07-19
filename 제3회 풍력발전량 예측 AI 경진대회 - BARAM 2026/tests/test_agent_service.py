from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_service.adapters import meta_gate_evaluation, phase_regime_evaluation
from agent_service.config import ServiceConfig
from agent_service.contracts import (
    Evaluation,
    Hypothesis,
    RunSpec,
    ValidationPlan,
    resolve_inside,
)
from agent_service.orchestrator import Orchestrator
from agent_service.policy import PromotionPolicy
from agent_service.runner import SafeModuleRunner
from agent_service.store import AgentStore
from agent_service.submission import AutomaticSubmissionController, CandidateValidator
from experiments.group3_rolling_catboost_blend import BlendPolicy, apply_policy


POLICY = {
    "version": "test-v1",
    "min_locked_score_delta": 0.00015,
    "min_locked_one_minus_nmae_delta": 0.0,
    "min_locked_ficr_delta": 0.0,
    "min_expected_macro_score_delta": 0.00005,
    "min_positive_month_fraction": 0.5,
    "min_bootstrap_positive_fraction": 0.8,
    "min_bootstrap_q05": -0.00025,
    "max_changed_ratio": 0.25,
    "max_p95_movement_ratio": 0.015,
    "failed_family_max_coverage_fraction": 0.25,
}


def make_config(root: Path) -> ServiceConfig:
    return ServiceConfig(
        root.resolve(),
        {
            "schema_version": 1,
            "database_path": "artifacts_final/agent_service/state.sqlite3",
            "run_root": "artifacts_final/agent_service/runs",
            "submission_dir": "submissions",
            "archive_dir": "submissions/archive",
            "allowed_module_prefixes": ["experiments."],
            "human_submission_required": True,
            "submission": {
                "auto_submit_enabled": False,
                "competition_id": "236727",
                "team_name_env": "DACON_TEAM_NAME",
                "token_env": "DACON_API_TOKEN",
                "max_daily_auto_submissions": 1,
                "max_total_auto_submissions": 5,
                "min_hours_between_submissions": 6,
                "max_file_bytes": 104857600,
                "memo_prefix": "test",
                "results_csv": "submissions/results.csv",
            },
            "policy": POLICY,
            "roles": {
                "research": "research",
                "modeling": "modeling",
                "validation": "validation",
                "steward": "steward",
                "leaderboard": "leaderboard",
                "orchestrator": "orchestrator",
            },
        },
    )


def passing_evaluation(family: str = "exact_oof_meta_gate") -> Evaluation:
    return Evaluation.from_dict(
        {
            "family": family,
            "locked_score_delta": 0.000232,
            "locked_one_minus_nmae_delta": 0.000092,
            "locked_ficr_delta": 0.000372,
            "expected_macro_score_delta": 0.000077,
            "positive_months": 4,
            "total_months": 6,
            "bootstrap_positive_fraction": 0.8965,
            "bootstrap_q05": -0.000076,
            "changed_ratio": 0.1106,
            "p95_movement_ratio": 0.0051,
        }
    )


def failing_phase_evaluation() -> Evaluation:
    return Evaluation.from_dict(
        {
            "family": "phase_regime_cross_group",
            "locked_score_delta": 0.000689,
            "locked_one_minus_nmae_delta": 0.000740,
            "locked_ficr_delta": 0.000637,
            "expected_macro_score_delta": 0.000230,
            "positive_months": 4,
            "total_months": 6,
            "bootstrap_positive_fraction": 0.649,
            "bootstrap_q05": -0.00243,
            "changed_ratio": 0.8578,
            "p95_movement_ratio": 0.01673,
        }
    )


def test_policy_passes_meta_gate_and_rejects_public_phase_pattern() -> None:
    policy = PromotionPolicy(POLICY)
    passed = policy.evaluate(passing_evaluation())
    failed = policy.evaluate(failing_phase_evaluation())
    assert passed.outcome == "candidate"
    assert passed.human_submission_required
    assert failed.outcome == "rejected"
    assert "candidate changes too many rows" in failed.reasons
    assert "day-bootstrap positive fraction is too low" in failed.reasons
    assert "day-bootstrap lower tail is too negative" in failed.reasons


def test_policy_can_require_nonnegative_worst_month() -> None:
    settings = {
        **POLICY,
        "require_worst_month_score_delta": True,
        "min_worst_month_score_delta": 0.0,
    }
    raw = passing_evaluation().to_dict()
    raw["worst_month_score_delta"] = -0.000001
    decision = PromotionPolicy(settings).evaluate(Evaluation.from_dict(raw))
    assert decision.outcome == "rejected"
    assert "worst-month score delta is negative" in decision.reasons

    raw["worst_month_score_delta"] = 0.0
    assert PromotionPolicy(settings).evaluate(Evaluation.from_dict(raw)).outcome == "candidate"


def test_policy_family_override_allows_structural_blend_coverage_only() -> None:
    settings = {
        **POLICY,
        "family_overrides": {
            "spatiotemporal_multitask_blend": {
                "max_changed_ratio": 1.0,
                "max_p95_movement_ratio": 0.04,
            }
        },
    }
    raw = passing_evaluation().to_dict()
    raw.update(
        {
            "family": "spatiotemporal_multitask_blend",
            "changed_ratio": 1.0,
            "p95_movement_ratio": 0.026,
        }
    )
    structural = Evaluation.from_dict(raw)
    assert PromotionPolicy(settings).evaluate(structural).outcome == "candidate"

    unrelated = Evaluation.from_dict({**raw, "family": "unrelated_family"})
    decision = PromotionPolicy(settings).evaluate(unrelated)
    assert decision.outcome == "rejected"
    assert "candidate changes too many rows" in decision.reasons


def test_rejected_run_is_not_selected_as_local_best(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Unstable local improvement",
                "family": "phase_regime_cross_group",
                "rationale": "Exercise selection safety after deterministic rejection.",
                "expected_signal": "A rejected run must not become local best.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.unstable_local",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
        )
    )

    decision = orchestrator.evaluate(run_id, failing_phase_evaluation())

    assert decision.outcome == "rejected"
    assert store.get_active_selection("baram_2026", "local_best") is None


def test_external_data_run_without_manifest_fails_closed(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Unproven external forecast",
                "family": "external_forecast_probe",
                "rationale": "Exercise mandatory external-data provenance.",
                "expected_signal": "A missing manifest must block promotion.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.external_probe",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
            tags={"external_data": "public_forecast"},
        )
    )
    decision = orchestrator.evaluate(
        run_id, passing_evaluation("external_forecast_probe")
    )
    assert decision.outcome == "rejected"
    assert "leakage risk is high" in decision.reasons
    assert "competition rule violation is present" in decision.reasons
    assert store.get_active_selection("baram_2026", "local_best") is None


def test_selection_can_be_deactivated_with_audit_event(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Safe local candidate",
                "family": "safe_local",
                "rationale": "Exercise audited selection deactivation.",
                "expected_signal": "A passing run becomes the local best.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.safe_local",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
        )
    )
    orchestrator.evaluate(run_id, passing_evaluation("safe_local"))
    assert store.get_active_selection("baram_2026", "local_best") is not None

    changed = store.deactivate_selection(
        "baram_2026", "local_best", "test cleanup", run_id=run_id
    )

    assert changed == 1
    assert store.get_active_selection("baram_2026", "local_best") is None


def test_failed_family_requires_75_percent_coverage_reduction() -> None:
    raw = passing_evaluation("phase_regime_cross_group").to_dict()
    raw["changed_ratio"] = 0.21
    decision = PromotionPolicy(POLICY).evaluate(
        Evaluation.from_dict(raw), latest_failed_family_coverage=0.80
    )
    assert decision.outcome == "rejected"
    assert any("75%" in reason for reason in decision.reasons)


def test_public_failure_guard_matches_family_group_and_direction() -> None:
    raw = passing_evaluation("phase_regime_cross_group_v2").to_dict()
    raw.update(
        {
            "family_group": "structural_cross_group",
            "direction": "forward_injection",
            "changed_ratio": 0.21,
        }
    )
    candidate = Evaluation.from_dict(raw)
    evidence = {
        "family_group": "structural_cross_group",
        "direction": "forward_injection",
        "changed_ratio": 0.80,
    }
    decision = PromotionPolicy(POLICY).evaluate(
        candidate, public_failure_evidence=evidence
    )
    assert decision.outcome == "rejected"
    assert any("method family/direction" in reason for reason in decision.reasons)

    # A differently-directed probe is not suppressed by the directional
    # evidence (the ordinary max_changed_ratio gate still applies).
    raw["direction"] = "reverse_injection"
    raw["changed_ratio"] = 0.20
    decision = PromotionPolicy(POLICY).evaluate(
        Evaluation.from_dict(raw), public_failure_evidence=evidence
    )
    assert decision.outcome == "candidate"


def test_run_spec_materializes_and_path_guard_rejects_escape(tmp_path: Path) -> None:
    spec = RunSpec.from_dict(
        {
            "hypothesis_id": 1,
            "module": "experiments.valid_run",
            "args": ["--output", "submissions/run{run_id}.csv"],
            "report_path": "artifacts_final/agent_service/runs/{run_id}/report.json",
            "evaluation_path": "artifacts_final/agent_service/runs/{run_id}/evaluation.json",
            "candidate_path": "submissions/run{run_id}.csv",
        }
    ).materialize(42)
    assert spec.candidate_path == "submissions/run42.csv"
    assert spec.report_path.endswith("/42/report.json")
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "../outside.json", "output")


def test_store_task_queue_and_public_family_feedback(tmp_path: Path) -> None:
    store = AgentStore(tmp_path / "state.sqlite3")
    store.initialize()
    task_id = store.create_task("research", {"topic": "wind"})
    task = store.claim_task("research")
    assert task is not None and task["id"] == task_id and task["status"] == "running"
    assert task["attempts"] == 1 and task["lease_until"]
    assert store.heartbeat_task(task_id, 60)
    assert store.claim_task("research") is None
    store.complete_task(task_id, {"hypothesis_id": 7})
    completed = store.list_rows("tasks")[0]
    assert completed["status"] == "completed"
    assert completed["payload"]["topic"] == "wind"
    assert completed["payload"]["result"]["hypothesis_id"] == 7
    store.record_public_result(
        {
            "family": "phase_regime_cross_group",
            "submission_id": 1494535,
            "file": "blend_best_phase_regime_crossg3.csv",
            "score": 0.6411378997,
            "one_minus_nmae": 0.8756126769,
            "ficr": 0.4066631226,
            "score_delta_vs_best": -0.0005174729,
            "changed_ratio": 0.8578,
            "submitted_at": "2026-07-17 18:11:40",
        }
    )
    assert store.latest_failed_family_coverage("phase_regime_cross_group") == pytest.approx(0.8578)

    store.record_public_result(
        {
            "family": "structural_variant",
            "family_group": "structural_cross_group",
            "direction": "forward_injection",
            "submission_id": 1494536,
            "file": "variant.csv",
            "score": 0.6409,
            "one_minus_nmae": 0.875,
            "ficr": 0.406,
            "score_delta_vs_best": -0.0007,
            "changed_ratio": 0.75,
            "submitted_at": "2026-07-17 19:11:40",
        }
    )
    evidence = store.latest_failed_family_evidence(
        "structural_cross_group", "forward_injection"
    )
    assert evidence is not None
    assert evidence["changed_ratio"] == pytest.approx(0.75)
    assert (
        store.latest_failed_family_evidence(
            "structural_cross_group", "reverse_injection"
        )
        is None
    )


def test_orchestrator_promotes_only_smaller_same_direction_public_probe(
    tmp_path: Path,
) -> None:
    settings = {**POLICY, "public_failure_guard": {
        "enabled": True,
        "require_same_direction": True,
        "max_coverage_fraction": 0.25,
    }}
    config = make_config(tmp_path)
    config = ServiceConfig(
        config.project_root,
        {**config.raw, "policy": settings},
    )
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    store.record_public_result(
        {
            "family": "seasonal_group3_affine",
            "family_group": "broad_positive_affine_transfer",
            "direction": "positive",
            "submission_id": 1999001,
            "file": "public_failed_affine.csv",
            "score": 0.60,
            "one_minus_nmae": 0.80,
            "ficr": 0.30,
            "score_delta_vs_best": -0.01,
            "changed_ratio": 0.75,
            "submitted_at": "2026-07-18 00:00:00",
        }
    )
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Same-direction affine probe",
                "family": "controlled_g12_scada_group3_monthmask_affine",
                "rationale": "Exercise public family/direction guard.",
                "expected_signal": "A high-coverage public loss must constrain variants.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.affine_probe",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
        )
    )
    raw = passing_evaluation("controlled_g12_scada_group3_monthmask_affine").to_dict()
    raw.update(
        {
            "family_group": "broad_positive_affine_transfer",
            "direction": "positive",
            "changed_ratio": 0.20,
        }
    )
    decision = orchestrator.evaluate(run_id, Evaluation.from_dict(raw))
    assert decision.outcome == "rejected"
    assert any("method family/direction" in reason for reason in decision.reasons)
    evidence = store.latest_failed_family_evidence(
        "broad_positive_affine_transfer", "positive"
    )
    assert evidence is not None and evidence["changed_ratio"] == pytest.approx(0.75)


def test_orchestrator_runs_safe_module_and_validates(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "experiments" / "dummy.py").write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--report', required=True)
p.add_argument('--evaluation', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
for value in (a.report,a.evaluation,a.output): Path(value).parent.mkdir(parents=True,exist_ok=True)
Path(a.report).write_text(json.dumps({'ok':True}),encoding='utf-8')
Path(a.evaluation).write_text(json.dumps({
'family':'safe_dummy','locked_score_delta':0.0003,'locked_one_minus_nmae_delta':0.0001,
'locked_ficr_delta':0.0002,'expected_macro_score_delta':0.0001,'positive_months':5,
'total_months':6,'bootstrap_positive_fraction':0.9,'bootstrap_q05':-0.0001,
'changed_ratio':0.1,'p95_movement_ratio':0.005}),encoding='utf-8')
Path(a.output).write_text('forecast_id,forecast_kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\\n',encoding='utf-8')
""",
        encoding="utf-8",
    )
    config = make_config(tmp_path)
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Safe dummy",
                "family": "safe_dummy",
                "rationale": "Exercise the control plane.",
                "expected_signal": "A deterministic positive result.",
            }
        )
    )
    spec = RunSpec.from_dict(
        {
            "hypothesis_id": hypothesis_id,
            "module": "experiments.dummy",
            "args": [
                "--report",
                "artifacts_final/agent_service/runs/{run_id}/report.json",
                "--evaluation",
                "artifacts_final/agent_service/runs/{run_id}/evaluation.json",
                "--output",
                "submissions/dummy_run{run_id}.csv",
            ],
            "report_path": "artifacts_final/agent_service/runs/{run_id}/report.json",
            "evaluation_path": "artifacts_final/agent_service/runs/{run_id}/evaluation.json",
            "candidate_path": "submissions/dummy_run{run_id}.csv",
        }
    )
    run_id = orchestrator.register_run(spec)
    result = orchestrator.execute(run_id)
    assert result.exit_code == 0 and result.expected_outputs_present
    decision = orchestrator.evaluate(run_id)
    assert decision.outcome == "candidate"
    assert store.get_run(run_id)["status"] == "candidate_pending_human"
    manifest = tmp_path / f"artifacts_final/agent_service/runs/{run_id}/run_manifest.json"
    assert manifest.is_file()


def test_runner_rejects_stale_expected_outputs(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "experiments" / "stale.py").write_text(
        "# Exit successfully without producing the declared outputs.\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    evaluation = tmp_path / "evaluation.json"
    report.write_text('{"old": true}', encoding="utf-8")
    evaluation.write_text('{"old": true}', encoding="utf-8")
    runner = SafeModuleRunner(make_config(tmp_path))
    result = runner.run(
        1,
        RunSpec(
            hypothesis_id=1,
            module="experiments.stale",
            args=(),
            report_path="report.json",
            evaluation_path="evaluation.json",
        ),
    )
    assert result.exit_code == 0
    assert not result.expected_outputs_present
    manifest = json.loads(
        (tmp_path / "artifacts_final/agent_service/runs/1/run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["output_fresh"] == {
        "report.json": False,
        "evaluation.json": False,
    }


def test_runner_rejects_external_candidate(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    runner = SafeModuleRunner(config)
    with pytest.raises(ValueError):
        runner.validate(
            RunSpec(
                hypothesis_id=1,
                module="experiments.good",
                args=(),
                report_path="report.json",
                evaluation_path="evaluation.json",
                candidate_path="outside.csv",
            )
        )


def test_report_adapters_emit_standard_contracts() -> None:
    phase = phase_regime_evaluation(
        {
            "locked_validation": {
                "metrics": {
                    "delta": {
                        "score": 0.000689,
                        "one_minus_nmae": 0.000740,
                        "ficr": 0.000637,
                    }
                },
                "expected_competition_macro_score_delta": 0.000230,
                "positive_months": 4,
                "monthly_deltas": {str(i): {} for i in range(6)},
                "day_bootstrap": {"positive_fraction": 0.649, "q05": -0.00243},
            },
            "final": {
                "changed_ratio": 0.8578,
                "p95_absolute_movement_kwh": 351.33,
            },
        }
    )
    meta = meta_gate_evaluation(
        {
            "validation": {
                "locked_h1_to_h2": {
                    "metrics": {
                        "delta": {
                            "score": 0.000232,
                            "one_minus_nmae": 0.000092,
                            "ficr": 0.000372,
                        }
                    }
                }
            },
            "locked_h2_day_bootstrap": {"positive_fraction": 0.8965, "q05": -0.000076},
            "locked_h2_monthly_deltas": {
                str(i): {"score": 0.001 if i < 4 else -0.001} for i in range(6)
            },
            "final": {
                "changed_ratio": 0.1106,
                "p95_absolute_movement_kwh": 106.24,
            },
        }
    )
    assert phase.family == "phase_regime_cross_group"
    assert meta.family == "exact_oof_meta_gate"
    assert meta.positive_months == 4


def _write_one_row_submission(root: Path, relative: str) -> Path:
    header = "forecast_id,forecast_kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"
    row = "TEST_0000,2025-01-01 01:00:00,1000,1000,1000\n"
    sample = root / "data" / "sample_submission.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(header + row, encoding="utf-8-sig")
    candidate = root / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(header + row, encoding="utf-8-sig")
    return candidate


def test_candidate_validator_and_submission_dry_run(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    candidate = _write_one_row_submission(tmp_path, "submissions/candidate.csv")
    audit = CandidateValidator(config).audit(candidate)
    assert audit.valid and audit.rows == 1

    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Submission dry run",
                "family": "safe_submit",
                "rationale": "Test guarded submission.",
                "expected_signal": "No external call in dry-run mode.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.safe_submit",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
            candidate_path="submissions/candidate.csv",
        )
    )
    assert orchestrator.evaluate(run_id, passing_evaluation("safe_submit")).outcome == "candidate"
    controller = AutomaticSubmissionController(config, store)
    result = controller.submit(run_id, execute=False)
    assert result["status"] == "dry_run_eligible"
    blocked = controller.submit(run_id, execute=True)
    assert blocked["status"] == "blocked"
    assert "auto_submit_enabled is false" in blocked["reasons"]


def test_leaderboard_sync_rejects_and_archives_scored_candidate(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    _write_one_row_submission(tmp_path, "submissions/scored.csv")
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Scored candidate",
                "family": "scored_family",
                "rationale": "Test public feedback.",
                "expected_signal": "Archive a public loss.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.scored",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
            candidate_path="submissions/scored.csv",
        )
    )
    orchestrator.evaluate(run_id, passing_evaluation("scored_family"))
    results = tmp_path / "submissions" / "results.csv"
    results.write_text(
        "submission_id,title,submitted_at,file,score,one_minus_nmae,ficr,notes\n"
        "1,best,2026-07-17 10:00:00,best.csv,0.642,0.876,0.408,best\n"
        "2,probe,2026-07-17 11:00:00,scored.csv,0.641,0.877,0.405,failed\n",
        encoding="utf-8-sig",
    )
    synced = orchestrator.sync_leaderboard()
    assert synced["count"] == 2
    assert store.get_run(run_id)["status"] == "archived"
    assert store.get_active_selection("baram_2026", "submission_candidate") is None
    assert store.get_active_selection("baram_2026", "local_best") is None
    assert not (tmp_path / "submissions" / "scored.csv").exists()
    assert (tmp_path / "submissions" / "archive" / "scored.csv").exists()


def _write_generic_competition_profile(root: Path) -> None:
    agents = root / ".agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "competition.json").write_text(
        """
{
  "slug":"generic_tabular","name":"Generic tabular","platform":"kaggle",
  "competition_id":"generic-tabular","task_type":"binary_classification",
  "metric_name":"auc","metric_direction":"maximize",
  "sample_submission_path":"data/sample_submission.csv",
  "id_columns":["id"],"target_columns":["probability"],
  "target_ranges":{"probability":{"min":0.0,"max":1.0}},
  "rules":{"remote_llm_raw_rows":"forbidden"}
}
""",
        encoding="utf-8",
    )


def test_validation_approval_tree_and_selection_are_separate(tmp_path: Path) -> None:
    _write_generic_competition_profile(tmp_path)
    base = make_config(tmp_path)
    base.raw["competition_profile_path"] = ".agents/competition.json"
    base.raw["governance"] = {
        "require_validation_approval": True,
        "require_submission_selection": True,
    }
    store = AgentStore(base.database_path)
    orchestrator = Orchestrator(base, store)
    orchestrator.initialize()
    plan_id = orchestrator.register_validation_plan(
        ValidationPlan.from_dict(
            {
                "competition_slug": "generic_tabular",
                "name": "grouped five fold",
                "method": "stratified_group_kfold",
                "rationale": "Keep repeated entities in one fold.",
                "n_splits": 5,
                "group_columns": ["customer_id"],
                "leakage_checks": ["duplicate entity audit"],
            }
        )
    )
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "competition_slug": "generic_tabular",
                "title": "CatBoost categorical handling",
                "family": "catboost_native",
                "rationale": "Avoid lossy ordinal encoding.",
                "expected_signal": "Higher stable OOF AUC.",
            }
        )
    )
    root_spec = RunSpec(
        hypothesis_id=hypothesis_id,
        module="experiments.catboost_native",
        args=(),
        report_path="artifacts_final/agent_service/runs/{run_id}/report.json",
        evaluation_path="artifacts_final/agent_service/runs/{run_id}/evaluation.json",
        validation_plan_id=plan_id,
    )
    with pytest.raises(ValueError, match="not been approved"):
        orchestrator.register_run(root_spec)
    orchestrator.approve(
        "validation_strategy",
        "validation_plan",
        plan_id,
        "approved",
        "human",
        "Group isolation and leakage checks are adequate.",
    )
    root_run = orchestrator.register_run(root_spec)
    with pytest.raises(ValueError, match="single bounded change"):
        RunSpec.from_dict({**root_spec.to_dict(), "parent_run_id": root_run})
    child_run = orchestrator.register_run(
        RunSpec.from_dict(
            {
                **root_spec.to_dict(),
                "parent_run_id": root_run,
                "change_summary": "increase depth from 6 to 7 only",
            }
        )
    )
    tree = store.experiment_tree()
    assert [row["depth"] for row in tree] == [0, 1]

    evaluation = Evaluation.from_dict(
        {**passing_evaluation("catboost_native").to_dict(), "selection_metric": 0.8431}
    )
    orchestrator.evaluate(root_run, evaluation)
    assert store.get_active_selection("generic_tabular", "local_best")["run_id"] == root_run
    assert store.get_active_selection("generic_tabular", "submission_candidate") is None
    orchestrator.evaluate(child_run, evaluation)
    orchestrator.select_run(
        child_run, "submission_candidate", "Diverse OOF member selected", "human"
    )
    assert (
        store.get_active_selection("generic_tabular", "submission_candidate")["run_id"]
        == child_run
    )


def test_competition_profile_drives_generic_submission_schema(tmp_path: Path) -> None:
    _write_generic_competition_profile(tmp_path)
    config = make_config(tmp_path)
    config.raw["competition_profile_path"] = ".agents/competition.json"
    sample = tmp_path / "data" / "sample_submission.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("id,probability\nA,0.5\n", encoding="utf-8-sig")
    candidate = tmp_path / "submissions" / "candidate.csv"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("id,probability\nA,0.9\n", encoding="utf-8-sig")
    assert CandidateValidator(config).audit(candidate).valid
    candidate.write_text("id,probability\nA,1.1\n", encoding="utf-8-sig")
    audit = CandidateValidator(config).audit(candidate)
    assert not audit.valid and "out-of-range probability" in audit.errors[0]


def test_automatic_mode_selects_only_a_promoted_run(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.raw["human_submission_required"] = False
    config.raw["governance"] = {"require_submission_selection": True}
    config.raw["submission"]["auto_submit_enabled"] = True
    _write_one_row_submission(tmp_path, "submissions/auto.csv")
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()
    hypothesis_id = orchestrator.propose(
        Hypothesis.from_dict(
            {
                "title": "Automatic guarded candidate",
                "family": "automatic_guard",
                "rationale": "Exercise the no-human deployment policy.",
                "expected_signal": "Eligible dry run only after promotion.",
            }
        )
    )
    run_id = orchestrator.register_run(
        RunSpec(
            hypothesis_id=hypothesis_id,
            module="experiments.auto_guard",
            args=(),
            report_path="artifacts_final/agent_service/runs/1/report.json",
            evaluation_path="artifacts_final/agent_service/runs/1/evaluation.json",
            candidate_path="submissions/auto.csv",
        )
    )
    orchestrator.evaluate(run_id, passing_evaluation("automatic_guard"))
    assert store.get_run(run_id)["status"] == "auto_submit_ready"
    selected = store.get_active_selection("baram_2026", "submission_candidate")
    assert selected is not None and selected["run_id"] == run_id
    assert orchestrator.submission_check(run_id)["status"] == "dry_run_eligible"
    blocked = orchestrator.submission_check(run_id, execute=True)
    assert blocked["status"] == "blocked"
    assert "environment variable is missing" in blocked["reasons"][0]


def test_rolling_catboost_policy_requires_unanimous_seed_direction() -> None:
    current = np.array([10_000.0, 10_000.0, 10_000.0])
    seeds = np.array(
        [
            [10_500.0, 10_400.0, 10_600.0],
            [10_500.0, 9_800.0, 10_300.0],
            [9_500.0, 9_600.0, 9_400.0],
        ]
    )
    policy = BlendPolicy(0.1, 0.0, 0.06, 0.02)
    candidate, gate = apply_policy(current, seeds, policy)
    assert gate.tolist() == [True, False, True]
    assert candidate[0] > current[0]
    assert candidate[2] < current[2]


def test_multitask_target_mask_normalizes_and_preserves_missing_labels() -> None:
    from experiments.group_multitask_catboost import make_multitask_targets

    labels = pd.DataFrame(
        {
            "kpx_group_1": [2160.0, 1080.0],
            "kpx_group_2": [4320.0, 2160.0],
            "kpx_group_3": [np.nan, 2100.0],
        }
    )
    target = make_multitask_targets(labels)

    assert np.isclose(target[0, 0], 0.10)
    assert np.isnan(target[1, 0])
    assert np.isclose(target[0, 1], 0.20)
    assert np.isnan(target[0, 2])
    assert np.isclose(target[1, 2], 0.10)
