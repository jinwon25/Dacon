from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_service.adapters import ADAPTERS, adapt_report
from agent_service.api import serve
from agent_service.config import load_config
from agent_service.contracts import Evaluation, Hypothesis, RunSpec, ValidationPlan
from agent_service.orchestrator import Orchestrator
from agent_service.store import AgentStore


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Competition Scientist experiment-agent control plane"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init")
    commands.add_parser("status")
    commands.add_parser("competition-show")

    validation = commands.add_parser("validation-add")
    validation.add_argument("json_path")

    approval = commands.add_parser("approval-record")
    approval.add_argument("json_path")

    hypothesis = commands.add_parser("hypothesis-add")
    hypothesis.add_argument("json_path")

    run_register = commands.add_parser("run-register")
    run_register.add_argument("json_path")

    run_execute = commands.add_parser("run-execute")
    run_execute.add_argument("run_id", type=int)

    run_evaluate = commands.add_parser("run-evaluate")
    run_evaluate.add_argument("run_id", type=int)
    evaluation_source = run_evaluate.add_mutually_exclusive_group()
    evaluation_source.add_argument("--evaluation")
    evaluation_source.add_argument(
        "--report", help="Adapt an experiment report and evaluate the run directly."
    )
    run_evaluate.add_argument("--adapter", choices=sorted(ADAPTERS))

    run_archive = commands.add_parser("run-archive")
    run_archive.add_argument("run_id", type=int)
    run_archive.add_argument("--apply", action="store_true")

    run_select = commands.add_parser("run-select")
    run_select.add_argument("run_id", type=int)
    run_select.add_argument(
        "selection_type", choices=["local_best", "submission_candidate", "ensemble_member"]
    )
    run_select.add_argument("--rationale", required=True)
    run_select.add_argument("--selected-by", default="human")

    commands.add_parser("tree")

    report_adapt = commands.add_parser("report-adapt")
    report_adapt.add_argument("adapter", choices=sorted(ADAPTERS))
    report_adapt.add_argument("input_path")
    report_adapt.add_argument("output_path")

    public = commands.add_parser("public-record")
    public.add_argument("json_path")

    submission = commands.add_parser("submission-check")
    submission.add_argument("run_id", type=int)
    submission.add_argument(
        "--execute",
        action="store_true",
        help="Call the official DACON API; requires config enablement and environment credentials.",
    )

    commands.add_parser("leaderboard-sync")

    cycle = commands.add_parser("auto-cycle")
    cycle.add_argument("--execute-submissions", action="store_true")

    cleanup = commands.add_parser("cleanup-artifacts")
    cleanup.add_argument("--apply", action="store_true", help="Move duplicate CSVs to archive/")

    task_add = commands.add_parser("task-add")
    task_add.add_argument("role")
    task_add.add_argument("json_path")
    task_add.add_argument("--run-id", type=int)

    task_claim = commands.add_parser("task-claim")
    task_claim.add_argument("role")
    task_claim.add_argument("--lease-seconds", type=int, default=900)

    task_heartbeat = commands.add_parser("task-heartbeat")
    task_heartbeat.add_argument("task_id", type=int)
    task_heartbeat.add_argument("--lease-seconds", type=int, default=900)

    task_complete = commands.add_parser("task-complete")
    task_complete.add_argument("task_id", type=int)
    task_complete.add_argument("json_path")

    listing = commands.add_parser("list")
    listing.add_argument(
        "table",
        choices=[
            "hypotheses",
            "competition_profiles",
            "validation_plans",
            "approvals",
            "runs",
            "run_attempts",
            "evaluations",
            "decisions",
            "public_results",
            "tasks",
            "submission_attempts",
            "selections",
            "events",
        ],
    )
    listing.add_argument("--limit", type=int, default=100)

    server = commands.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    # Pure report conversion deliberately avoids opening or initializing the run
    # database. This makes it safe to prepare an Evaluation before registering a run.
    if args.command == "report-adapt":
        evaluation = adapt_report(args.adapter, _read_json(args.input_path))
        _write_json(args.output_path, evaluation.to_dict())
        print(
            json.dumps(
                {"output": args.output_path, "evaluation": evaluation.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    config = load_config(root, Path(args.config) if args.config else None)
    store = AgentStore(config.database_path)
    orchestrator = Orchestrator(config, store)
    orchestrator.initialize()

    if args.command == "init":
        output: Any = {
            "status": "initialized",
            "database": str(config.database_path),
            "run_root": str(config.run_root),
        }
    elif args.command == "status":
        output = orchestrator.status()
    elif args.command == "competition-show":
        output = config.competition_profile.to_dict()
    elif args.command == "validation-add":
        output = {
            "validation_plan_id": orchestrator.register_validation_plan(
                ValidationPlan.from_dict(_read_json(args.json_path))
            )
        }
    elif args.command == "approval-record":
        payload = _read_json(args.json_path)
        output = {
            "approval_id": orchestrator.approve(
                str(payload["gate_type"]),
                str(payload["subject_type"]),
                int(payload["subject_id"]),
                str(payload["decision"]),
                str(payload["reviewer"]),
                str(payload["reason"]),
            )
        }
    elif args.command == "hypothesis-add":
        output = {"hypothesis_id": orchestrator.propose(Hypothesis.from_dict(_read_json(args.json_path)))}
    elif args.command == "run-register":
        output = {"run_id": orchestrator.register_run(RunSpec.from_dict(_read_json(args.json_path)))}
    elif args.command == "run-execute":
        output = orchestrator.execute(args.run_id).to_dict()
    elif args.command == "run-evaluate":
        if args.report and not args.adapter:
            raise ValueError("--report requires --adapter")
        if args.adapter and not args.report:
            raise ValueError("--adapter requires --report")
        if args.report:
            evaluation = adapt_report(args.adapter, _read_json(args.report))
        else:
            evaluation = (
                Evaluation.from_dict(_read_json(args.evaluation))
                if args.evaluation
                else None
            )
        output = orchestrator.evaluate(args.run_id, evaluation).to_dict()
    elif args.command == "run-archive":
        output = orchestrator.archive_rejected(args.run_id, apply=args.apply)
    elif args.command == "run-select":
        output = {
            "selection_id": orchestrator.select_run(
                args.run_id,
                args.selection_type,
                args.rationale,
                args.selected_by,
            )
        }
    elif args.command == "tree":
        output = store.experiment_tree()
    elif args.command == "public-record":
        orchestrator.record_public_result(_read_json(args.json_path))
        output = {"status": "recorded"}
    elif args.command == "submission-check":
        output = orchestrator.submission_check(args.run_id, execute=args.execute)
    elif args.command == "leaderboard-sync":
        output = orchestrator.sync_leaderboard()
    elif args.command == "auto-cycle":
        output = orchestrator.auto_cycle(
            execute_submissions=args.execute_submissions
        )
    elif args.command == "cleanup-artifacts":
        output = orchestrator.cleanup_submission_artifacts(apply=args.apply)
    elif args.command == "task-add":
        if args.role not in config.roles:
            raise ValueError(f"Unknown role: {args.role}")
        output = {
            "task_id": store.create_task(
                args.role, _read_json(args.json_path), run_id=args.run_id
            )
        }
    elif args.command == "task-claim":
        output = {"task": store.claim_task(args.role, args.lease_seconds)}
    elif args.command == "task-heartbeat":
        output = {
            "task_id": args.task_id,
            "lease_until": store.heartbeat_task(args.task_id, args.lease_seconds),
        }
    elif args.command == "task-complete":
        store.complete_task(args.task_id, _read_json(args.json_path))
        output = {"task_id": args.task_id, "status": "completed"}
    elif args.command == "list":
        output = store.list_rows(args.table, args.limit)
    elif args.command == "serve":
        serve(
            orchestrator,
            host=args.host,
            port=args.port,
            token=os.environ.get("BARAM_AGENT_TOKEN"),
        )
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps(output, ensure_ascii=False, indent=2))
