from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agent_service.contracts import Evaluation, Hypothesis, RunSpec, ValidationPlan
from agent_service.orchestrator import Orchestrator


MAX_BODY_BYTES = 1_048_576


def make_handler(orchestrator: Orchestrator, token: str | None):
    class AgentRequestHandler(BaseHTTPRequestHandler):
        server_version = "CompetitionScientist/1.0"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _authorized(self) -> bool:
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            return hmac.compare_digest(supplied, expected)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("JSON body size is invalid")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _guard(self) -> bool:
            if self._authorized():
                return True
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

        def do_GET(self) -> None:
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                if path == "/health":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "service": "competition-scientist",
                            "human_submission_required": orchestrator.config.human_submission_required,
                        },
                    )
                    return
                if path == "/v1/status":
                    self._json(HTTPStatus.OK, orchestrator.status())
                    return
                if path == "/v1/tree":
                    self._json(HTTPStatus.OK, orchestrator.store.experiment_tree())
                    return
                if path.startswith("/v1/"):
                    table = path.removeprefix("/v1/")
                    self._json(HTTPStatus.OK, orchestrator.store.list_rows(table))
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except (ValueError, KeyError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_POST(self) -> None:
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/v1/hypotheses":
                    identifier = orchestrator.propose(Hypothesis.from_dict(body))
                    self._json(HTTPStatus.CREATED, {"hypothesis_id": identifier})
                    return
                if path == "/v1/validation-plans":
                    identifier = orchestrator.register_validation_plan(
                        ValidationPlan.from_dict(body)
                    )
                    self._json(
                        HTTPStatus.CREATED, {"validation_plan_id": identifier}
                    )
                    return
                if path == "/v1/approvals":
                    identifier = orchestrator.approve(
                        str(body["gate_type"]),
                        str(body["subject_type"]),
                        int(body["subject_id"]),
                        str(body["decision"]),
                        str(body["reviewer"]),
                        str(body["reason"]),
                    )
                    self._json(HTTPStatus.CREATED, {"approval_id": identifier})
                    return
                if path == "/v1/runs":
                    identifier = orchestrator.register_run(RunSpec.from_dict(body))
                    self._json(HTTPStatus.CREATED, {"run_id": identifier})
                    return
                if path.startswith("/v1/runs/") and path.endswith("/evaluate"):
                    run_id = int(path.split("/")[3])
                    decision = orchestrator.evaluate(
                        run_id, Evaluation.from_dict(body)
                    )
                    self._json(HTTPStatus.OK, decision.to_dict())
                    return
                if path == "/v1/selections":
                    identifier = orchestrator.select_run(
                        int(body["run_id"]),
                        str(body["selection_type"]),
                        str(body["rationale"]),
                        str(body.get("selected_by", "human")),
                    )
                    self._json(HTTPStatus.CREATED, {"selection_id": identifier})
                    return
                if path == "/v1/tasks/claim":
                    task = orchestrator.store.claim_task(
                        str(body["role"]), int(body.get("lease_seconds", 900))
                    )
                    self._json(HTTPStatus.OK, {"task": task})
                    return
                if path.startswith("/v1/tasks/") and path.endswith("/heartbeat"):
                    task_id = int(path.split("/")[3])
                    lease_until = orchestrator.store.heartbeat_task(
                        task_id, int(body.get("lease_seconds", 900))
                    )
                    self._json(
                        HTTPStatus.OK,
                        {"task_id": task_id, "lease_until": lease_until},
                    )
                    return
                if path.startswith("/v1/tasks/") and path.endswith("/complete"):
                    task_id = int(path.split("/")[3])
                    orchestrator.store.complete_task(task_id, body.get("result", body))
                    self._json(HTTPStatus.OK, {"task_id": task_id, "status": "completed"})
                    return
                if path == "/v1/public-results":
                    orchestrator.record_public_result(body)
                    self._json(HTTPStatus.CREATED, {"status": "recorded"})
                    return
                if path == "/v1/auto-cycle":
                    # Network callers may inspect eligibility and sync scores, but cannot
                    # trigger the external submission side effect.
                    self._json(HTTPStatus.OK, orchestrator.auto_cycle(False))
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return AgentRequestHandler


def serve(
    orchestrator: Orchestrator,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("A bearer token is required when binding outside localhost")
    orchestrator.initialize()
    server = ThreadingHTTPServer((host, port), make_handler(orchestrator, token))
    print(f"Competition Scientist listening on http://{host}:{port}", flush=True)
    server.serve_forever()
