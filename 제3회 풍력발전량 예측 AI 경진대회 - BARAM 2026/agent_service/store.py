from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from agent_service.contracts import (
    CompetitionProfile,
    Evaluation,
    Hypothesis,
    RunSpec,
    ValidationPlan,
)
from agent_service.policy import PolicyDecision


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStore:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    family TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS competition_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS validation_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition_slug TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
                    module TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    exit_code INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id),
                    family TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    outcome TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS public_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER REFERENCES runs(id),
                    family TEXT NOT NULL,
                    submission_id INTEGER NOT NULL UNIQUE,
                    file TEXT NOT NULL,
                    score REAL NOT NULL,
                    one_minus_nmae REAL NOT NULL,
                    ficr REAL NOT NULL,
                    score_delta_vs_best REAL NOT NULL,
                    changed_ratio REAL,
                    submitted_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    run_id INTEGER REFERENCES runs(id),
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    lease_until TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submission_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    file TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    UNIQUE(file_sha256, dry_run)
                );
                CREATE TABLE IF NOT EXISTS run_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    attempt_no INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    manifest_path TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    UNIQUE(run_id, attempt_no)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER REFERENCES runs(id),
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gate_type TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS selections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition_slug TEXT NOT NULL,
                    selection_type TEXT NOT NULL,
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    active INTEGER NOT NULL DEFAULT 1,
                    rationale TEXT NOT NULL,
                    selected_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(connection, "tasks", "lease_until", "TEXT")
            self._ensure_column(
                connection, "tasks", "attempts", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "runs", "parent_run_id", "INTEGER")
            self._ensure_column(connection, "runs", "validation_plan_id", "INTEGER")
            # Public evidence predates the generalized family/direction guard;
            # keep the migration additive so existing SQLite state remains
            # readable and old rows continue to use their concrete family.
            self._ensure_column(connection, "public_results", "family_group", "TEXT")
            self._ensure_column(connection, "public_results", "direction", "TEXT")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def register_competition(self, profile: CompetitionProfile) -> None:
        timestamp = utc_now()
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM competition_profiles WHERE slug=?",
                (profile.slug,),
            ).fetchone()
            if existing is not None and existing["payload_json"] == payload:
                return
            connection.execute(
                """
                INSERT INTO competition_profiles(slug,payload_json,created_at,updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(slug) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (profile.slug, payload, timestamp, timestamp),
            )
            self._event(connection, "competition.registered", profile.to_dict())

    def create_validation_plan(self, plan: ValidationPlan) -> int:
        timestamp = utc_now()
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM competition_profiles WHERE slug=?",
                (plan.competition_slug,),
            ).fetchone() is None:
                raise KeyError(f"Unknown competition: {plan.competition_slug}")
            cursor = connection.execute(
                """
                INSERT INTO validation_plans(
                    competition_slug,payload_json,created_at,updated_at
                ) VALUES(?,?,?,?)
                """,
                (
                    plan.competition_slug,
                    json.dumps(plan.to_dict(), ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            plan_id = int(cursor.lastrowid)
            self._event(
                connection,
                "validation_plan.created",
                {"validation_plan_id": plan_id, **plan.to_dict()},
            )
            return plan_id

    def record_approval(
        self,
        gate_type: str,
        subject_type: str,
        subject_id: int,
        decision: str,
        reviewer: str,
        reason: str,
    ) -> int:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer and reason are required")
        with self.connect() as connection:
            if subject_type == "validation_plan" and connection.execute(
                "SELECT 1 FROM validation_plans WHERE id=?", (int(subject_id),)
            ).fetchone() is None:
                raise KeyError(f"Unknown validation plan: {subject_id}")
            cursor = connection.execute(
                """
                INSERT INTO approvals(
                    gate_type,subject_type,subject_id,decision,reviewer,reason,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    gate_type,
                    subject_type,
                    int(subject_id),
                    decision,
                    reviewer.strip(),
                    reason.strip(),
                    utc_now(),
                ),
            )
            if subject_type == "validation_plan":
                connection.execute(
                    "UPDATE validation_plans SET status=?,updated_at=? WHERE id=?",
                    (decision, utc_now(), int(subject_id)),
                )
            approval_id = int(cursor.lastrowid)
            self._event(
                connection,
                "approval.recorded",
                {
                    "approval_id": approval_id,
                    "gate_type": gate_type,
                    "subject_type": subject_type,
                    "subject_id": int(subject_id),
                    "decision": decision,
                    "reviewer": reviewer.strip(),
                    "reason": reason.strip(),
                },
            )
            return approval_id

    def is_approved(self, gate_type: str, subject_type: str, subject_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT decision FROM approvals
                WHERE gate_type=? AND subject_type=? AND subject_id=?
                ORDER BY id DESC LIMIT 1
                """,
                (gate_type, subject_type, int(subject_id)),
            ).fetchone()
            return row is not None and row["decision"] == "approved"

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        payload: dict[str, Any],
        run_id: int | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
            (run_id, event_type, json.dumps(payload, ensure_ascii=False), utc_now()),
        )

    def create_hypothesis(self, hypothesis: Hypothesis) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO hypotheses(title,family,payload_json,created_at) VALUES(?,?,?,?)",
                (
                    hypothesis.title,
                    hypothesis.family,
                    json.dumps(hypothesis.to_dict(), ensure_ascii=False),
                    utc_now(),
                ),
            )
            hypothesis_id = int(cursor.lastrowid)
            self._event(connection, "hypothesis.created", hypothesis.to_dict())
            return hypothesis_id

    def create_run(self, spec: RunSpec) -> int:
        timestamp = utc_now()
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM hypotheses WHERE id=?", (spec.hypothesis_id,)
            ).fetchone() is None:
                raise KeyError(f"Unknown hypothesis: {spec.hypothesis_id}")
            if spec.parent_run_id is not None and connection.execute(
                "SELECT 1 FROM runs WHERE id=?", (spec.parent_run_id,)
            ).fetchone() is None:
                raise KeyError(f"Unknown parent run: {spec.parent_run_id}")
            if spec.validation_plan_id is not None and connection.execute(
                "SELECT 1 FROM validation_plans WHERE id=?", (spec.validation_plan_id,)
            ).fetchone() is None:
                raise KeyError(f"Unknown validation plan: {spec.validation_plan_id}")
            cursor = connection.execute(
                """
                INSERT INTO runs(
                    hypothesis_id,module,spec_json,parent_run_id,validation_plan_id,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    spec.hypothesis_id,
                    spec.module,
                    json.dumps(spec.to_dict(), ensure_ascii=False),
                    spec.parent_run_id,
                    spec.validation_plan_id,
                    timestamp,
                    timestamp,
                ),
            )
            run_id = int(cursor.lastrowid)
            self._event(connection, "run.created", spec.to_dict(), run_id)
            return run_id

    def select_run(
        self,
        competition_slug: str,
        selection_type: str,
        run_id: int,
        rationale: str,
        selected_by: str,
    ) -> int:
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM runs WHERE id=?", (int(run_id),)
            ).fetchone() is None:
                raise KeyError(f"Unknown run: {run_id}")
            connection.execute(
                """
                UPDATE selections SET active=0
                WHERE competition_slug=? AND selection_type=? AND active=1
                """,
                (competition_slug, selection_type),
            )
            cursor = connection.execute(
                """
                INSERT INTO selections(
                    competition_slug,selection_type,run_id,rationale,selected_by,created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    competition_slug,
                    selection_type,
                    int(run_id),
                    rationale.strip(),
                    selected_by.strip(),
                    utc_now(),
                ),
            )
            selection_id = int(cursor.lastrowid)
            self._event(
                connection,
                "run.selected",
                {
                    "selection_id": selection_id,
                    "competition_slug": competition_slug,
                    "selection_type": selection_type,
                    "run_id": int(run_id),
                    "rationale": rationale.strip(),
                    "selected_by": selected_by.strip(),
                },
                int(run_id),
            )
            return selection_id

    def get_active_selection(
        self, competition_slug: str, selection_type: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM selections
                WHERE competition_slug=? AND selection_type=? AND active=1
                ORDER BY id DESC LIMIT 1
                """,
                (competition_slug, selection_type),
            ).fetchone()
            return None if row is None else dict(row)

    def deactivate_selection(
        self,
        competition_slug: str,
        selection_type: str,
        reason: str,
        run_id: int | None = None,
    ) -> int:
        reason = reason.strip()
        if not reason:
            raise ValueError("selection deactivation requires a reason")
        with self.connect() as connection:
            parameters: list[Any] = [competition_slug, selection_type]
            run_clause = ""
            if run_id is not None:
                run_clause = " AND run_id=?"
                parameters.append(int(run_id))
            cursor = connection.execute(
                """
                UPDATE selections SET active=0
                WHERE competition_slug=? AND selection_type=? AND active=1
                """
                + run_clause,
                parameters,
            )
            changed = int(cursor.rowcount)
            if changed:
                self._event(
                    connection,
                    "selection.deactivated",
                    {
                        "competition_slug": competition_slug,
                        "selection_type": selection_type,
                        "run_id": run_id,
                        "reason": reason,
                        "changed": changed,
                    },
                    run_id,
                )
            return changed

    def experiment_tree(self, limit: int = 10_000) -> list[dict[str, Any]]:
        rows = list(reversed(self.list_rows("runs", limit=limit)))
        depth_by_id: dict[int, int] = {}
        for row in rows:
            parent = row.get("parent_run_id")
            depth_by_id[int(row["id"])] = 0 if parent is None else depth_by_id.get(int(parent), 0) + 1
            row["depth"] = depth_by_id[int(row["id"])]
        return rows

    def get_run(self, run_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            output = dict(row)
            output["spec"] = json.loads(output.pop("spec_json"))
            return output

    def set_run_status(self, run_id: int, status: str, exit_code: int | None = None) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status=?,exit_code=COALESCE(?,exit_code),updated_at=? WHERE id=?",
                (status, exit_code, utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown run: {run_id}")
            self._event(
                connection,
                "run.status",
                {"status": status, "exit_code": exit_code},
                run_id,
            )

    def start_run_attempt(self, run_id: int) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_no),0) AS n FROM run_attempts WHERE run_id=?",
                (run_id,),
            ).fetchone()
            attempt_no = int(row["n"]) + 1
            connection.execute(
                "INSERT INTO run_attempts(run_id,attempt_no,status,started_at) VALUES(?,?,?,?)",
                (run_id, attempt_no, "running", utc_now()),
            )
            self._event(
                connection,
                "run.attempt_started",
                {"attempt_no": attempt_no},
                run_id,
            )
            return attempt_no

    def finish_run_attempt(
        self, run_id: int, attempt_no: int, status: str, manifest_path: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE run_attempts SET status=?,manifest_path=?,finished_at=?
                WHERE run_id=? AND attempt_no=?
                """,
                (status, manifest_path, utc_now(), run_id, attempt_no),
            )
            self._event(
                connection,
                "run.attempt_finished",
                {
                    "attempt_no": attempt_no,
                    "status": status,
                    "manifest_path": manifest_path,
                },
                run_id,
            )

    def record_evaluation(self, run_id: int, evaluation: Evaluation) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evaluations(run_id,family,payload_json,created_at) VALUES(?,?,?,?)",
                (
                    run_id,
                    evaluation.family,
                    json.dumps(evaluation.to_dict(), ensure_ascii=False),
                    utc_now(),
                ),
            )
            self._event(connection, "evaluation.recorded", evaluation.to_dict(), run_id)

    def record_decision(self, run_id: int, decision: PolicyDecision) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO decisions(run_id,outcome,policy_version,payload_json,created_at) VALUES(?,?,?,?,?)",
                (
                    run_id,
                    decision.outcome,
                    decision.policy_version,
                    json.dumps(decision.to_dict(), ensure_ascii=False),
                    utc_now(),
                ),
            )
            self._event(connection, "decision.recorded", decision.to_dict(), run_id)

    def get_evaluation(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM evaluations WHERE run_id=?", (run_id,)
            ).fetchone()
            return None if row is None else json.loads(row["payload_json"])

    def get_latest_decision(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM decisions WHERE run_id=? ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            return None if row is None else json.loads(row["payload_json"])

    def latest_failed_family_coverage(self, family: str) -> float | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT changed_ratio FROM public_results
                WHERE family=? AND score_delta_vs_best<0 AND changed_ratio IS NOT NULL
                ORDER BY id DESC LIMIT 1
                """,
                (family,),
            ).fetchone()
            return None if row is None else float(row["changed_ratio"])

    def latest_failed_family_evidence(
        self, family_group: str, direction: str = "unknown"
    ) -> dict[str, Any] | None:
        """Return the latest public loss for a method family/group.

        A direction match is applied when both the candidate and public row
        provide one.  Unknown directions are retained for reporting but are
        not used to block a differently-labelled candidate; this prevents old
        rows from accidentally suppressing unrelated experiments.
        """
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT family, family_group, direction, changed_ratio,
                       score_delta_vs_best, submission_id, file
                FROM public_results
                WHERE score_delta_vs_best<0
                  AND changed_ratio IS NOT NULL
                  AND (family_group=? OR (family_group IS NULL AND family=?))
                ORDER BY id DESC
                """,
                (family_group, family_group),
            ).fetchall()
            candidate_direction = direction.strip().lower() or "unknown"
            row = next(
                (
                    item
                    for item in rows
                    if candidate_direction != "unknown"
                    and (item["direction"] or "unknown").strip().lower()
                    == candidate_direction
                ),
                None,
            )
            if row is None:
                return None
            return {
                "family": row["family"],
                "family_group": row["family_group"] or row["family"],
                "direction": (row["direction"] or "unknown").lower(),
                "changed_ratio": float(row["changed_ratio"]),
                "score_delta_vs_best": float(row["score_delta_vs_best"]),
                "submission_id": int(row["submission_id"]),
                "file": row["file"],
            }

    def submission_budget(self, day_start_utc: str, day_end_utc: str) -> dict[str, Any]:
        with self.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM submission_attempts WHERE dry_run=0"
            ).fetchone()["n"]
            daily = connection.execute(
                """
                SELECT COUNT(*) AS n FROM submission_attempts
                WHERE dry_run=0 AND attempted_at>=? AND attempted_at<?
                """,
                (day_start_utc, day_end_utc),
            ).fetchone()["n"]
            latest = connection.execute(
                "SELECT attempted_at FROM submission_attempts WHERE dry_run=0 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return {
                "total": int(total),
                "daily": int(daily),
                "latest_attempted_at": None if latest is None else latest["attempted_at"],
            }

    def has_submission_hash(self, file_sha256: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM submission_attempts WHERE file_sha256=? AND dry_run=0",
                (file_sha256,),
            ).fetchone()
            return row is not None

    def record_submission_attempt(
        self,
        run_id: int,
        file: str,
        file_sha256: str,
        dry_run: bool,
        status: str,
        response: dict[str, Any],
    ) -> None:
        timestamp = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO submission_attempts(
                    run_id,file,file_sha256,dry_run,status,response_json,attempted_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(file_sha256,dry_run) DO UPDATE SET
                    status=excluded.status,
                    response_json=excluded.response_json,
                    attempted_at=excluded.attempted_at
                """,
                (
                    run_id,
                    file,
                    file_sha256,
                    int(dry_run),
                    status,
                    json.dumps(response, ensure_ascii=False),
                    timestamp,
                ),
            )
            self._event(
                connection,
                "submission.attempted",
                {
                    "file": file,
                    "file_sha256": file_sha256,
                    "dry_run": dry_run,
                    "status": status,
                    "response": response,
                },
                run_id,
            )

    def record_public_result(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO public_results(
                    run_id,family,submission_id,file,score,one_minus_nmae,ficr,
                    score_delta_vs_best,changed_ratio,submitted_at,created_at,
                    family_group,direction
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    family=excluded.family,
                    file=excluded.file,
                    score=excluded.score,
                    one_minus_nmae=excluded.one_minus_nmae,
                    ficr=excluded.ficr,
                    score_delta_vs_best=excluded.score_delta_vs_best,
                    changed_ratio=excluded.changed_ratio,
                    submitted_at=excluded.submitted_at,
                    created_at=excluded.created_at,
                    family_group=excluded.family_group,
                    direction=excluded.direction
                """,
                (
                    payload.get("run_id"),
                    payload["family"],
                    int(payload["submission_id"]),
                    payload["file"],
                    float(payload["score"]),
                    float(payload["one_minus_nmae"]),
                    float(payload["ficr"]),
                    float(payload["score_delta_vs_best"]),
                    (
                        None
                        if payload.get("changed_ratio") is None
                        else float(payload["changed_ratio"])
                    ),
                    payload["submitted_at"],
                    utc_now(),
                    payload.get("family_group") or payload["family"],
                    str(payload.get("direction", "unknown")).strip().lower() or "unknown",
                ),
            )
            run_id = payload.get("run_id")
            self._event(connection, "public_result.recorded", payload, run_id)

    def create_task(
        self, role: str, payload: dict[str, Any], run_id: int | None = None
    ) -> int:
        timestamp = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(role,run_id,payload_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (role, run_id, json.dumps(payload, ensure_ascii=False), timestamp, timestamp),
            )
            task_id = int(cursor.lastrowid)
            self._event(
                connection,
                "task.created",
                {"task_id": task_id, "role": role, "payload": payload},
                run_id,
            )
            return task_id

    def claim_task(self, role: str, lease_seconds: int = 900) -> dict[str, Any] | None:
        if not 30 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 30 and 86400")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE tasks SET status='queued',lease_until=NULL
                WHERE status='running' AND lease_until IS NOT NULL AND lease_until<?
                """,
                (now,),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE role=? AND status='queued' ORDER BY id LIMIT 1",
                (role,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE tasks SET status='running',attempts=attempts+1,lease_until=?,updated_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat(),
                    now,
                    row["id"],
                ),
            )
            output = dict(row)
            output["status"] = "running"
            output["attempts"] = int(output["attempts"]) + 1
            output["lease_until"] = (
                datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            ).isoformat()
            output["payload"] = json.loads(output.pop("payload_json"))
            self._event(
                connection,
                "task.claimed",
                {"task_id": output["id"], "role": role},
                output["run_id"],
            )
            return output

    def heartbeat_task(self, task_id: int, lease_seconds: int = 900) -> str:
        if not 30 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 30 and 86400")
        lease_until = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET lease_until=?,updated_at=?
                WHERE id=? AND status='running'
                """,
                (lease_until, utc_now(), task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("task is not running or does not exist")
            self._event(
                connection,
                "task.heartbeat",
                {"task_id": task_id, "lease_until": lease_until},
            )
        return lease_until

    def complete_task(self, task_id: int, result: dict[str, Any]) -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT run_id,payload_json FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            payload = json.loads(row["payload_json"])
            payload["result"] = result
            connection.execute(
                """
                UPDATE tasks SET status='completed',payload_json=?,lease_until=NULL,updated_at=?
                WHERE id=?
                """,
                (json.dumps(payload, ensure_ascii=False), utc_now(), task_id),
            )
            self._event(
                connection,
                "task.completed",
                {"task_id": task_id, "result": result},
                row["run_id"],
            )

    def list_rows(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        allowed = {
            "competition_profiles",
            "validation_plans",
            "approvals",
            "hypotheses",
            "runs",
            "run_attempts",
            "evaluations",
            "decisions",
            "public_results",
            "tasks",
            "submission_attempts",
            "selections",
            "events",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported table: {table}")
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
            output = []
            for row in rows:
                item = dict(row)
                for key in tuple(item):
                    if key.endswith("_json"):
                        item[key[:-5]] = json.loads(item.pop(key))
                output.append(item)
            return output
