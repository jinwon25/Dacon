from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_service.config import ServiceConfig
from agent_service.contracts import RunSpec, resolve_inside
from agent_service.submission import sha256_file


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    timed_out: bool
    duration_seconds: float
    command: tuple[str, ...]
    stdout_path: str
    stderr_path: str
    manifest_path: str
    expected_outputs_present: bool

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["command"] = list(self.command)
        return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class SafeModuleRunner:
    """Run registered experiment modules without invoking a shell."""

    def __init__(self, config: ServiceConfig):
        self.config = config

    def validate(self, spec: RunSpec) -> None:
        if not spec.module.startswith(self.config.allowed_module_prefixes):
            raise ValueError(f"Module is not allowed: {spec.module}")
        resolve_inside(self.config.project_root, spec.report_path, "report_path")
        resolve_inside(self.config.project_root, spec.evaluation_path, "evaluation_path")
        if spec.candidate_path:
            candidate = resolve_inside(
                self.config.project_root, spec.candidate_path, "candidate_path"
            )
            try:
                candidate.relative_to(self.config.submission_dir)
            except ValueError as exc:
                raise ValueError("candidate_path must stay inside submissions/") from exc
        for input_path in spec.input_paths:
            path = resolve_inside(self.config.project_root, input_path, "input_path")
            if not path.is_file():
                raise ValueError(f"Declared input file is missing: {input_path}")
        for argument in spec.args:
            if argument in {"..", "."} or "../" in argument.replace("\\", "/"):
                raise ValueError(f"Unsafe argument path traversal: {argument}")
            path = Path(argument)
            if path.is_absolute():
                try:
                    path.resolve().relative_to(self.config.project_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Absolute argument must stay inside project root: {argument}"
                    ) from exc

    def _git_state(self) -> dict[str, Any]:
        def call(*args: str) -> str | None:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            return result.stdout.strip() if result.returncode == 0 else None

        status = call("status", "--porcelain")
        return {
            "commit": call("rev-parse", "HEAD"),
            "branch": call("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
        }

    def run(self, run_id: int, spec: RunSpec, attempt_no: int = 1) -> RunResult:
        spec = spec.materialize(run_id)
        self.validate(spec)
        run_dir = self.config.run_root / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = run_dir / f"attempt_{attempt_no:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        attempt_manifest_path = attempt_dir / "run_manifest.json"
        manifest_path = run_dir / "run_manifest.json"
        command = (sys.executable, "-B", "-m", spec.module, *spec.args)
        expected = [
            resolve_inside(self.config.project_root, spec.report_path, "report_path"),
            resolve_inside(
                self.config.project_root, spec.evaluation_path, "evaluation_path"
            ),
        ]
        if spec.candidate_path:
            expected.append(
                resolve_inside(
                    self.config.project_root, spec.candidate_path, "candidate_path"
                )
            )
        before_outputs = {
            path: (
                path.stat().st_mtime_ns,
                path.stat().st_size,
                sha256_file(path),
            )
            for path in expected
            if path.is_file()
        }
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.setdefault("PYTHONHASHSEED", "0")
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.project_root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=spec.timeout_seconds,
                check=False,
                shell=False,
            )
            exit_code = int(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            stderr += f"\nTimed out after {spec.timeout_seconds} seconds."
        duration = time.monotonic() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        outputs_fresh = {
            str(path.relative_to(self.config.project_root)): bool(
                path.is_file()
                and (
                    path not in before_outputs
                    or (
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                        sha256_file(path),
                    )
                    != before_outputs[path]
                )
            )
            for path in expected
        }
        # A successful process may not reuse an artifact left by an earlier
        # attempt.  Every declared output must be created or rewritten now.
        outputs_present = all(outputs_fresh.values())
        input_fingerprints = {
            path: sha256_file(resolve_inside(self.config.project_root, path, "input_path"))
            for path in spec.input_paths
        }
        output_fingerprints = {
            str(path.relative_to(self.config.project_root)): sha256_file(path)
            for path in expected
            if path.is_file()
        }
        result = RunResult(
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration,
            command=command,
            stdout_path=str(stdout_path.relative_to(self.config.project_root)),
            stderr_path=str(stderr_path.relative_to(self.config.project_root)),
            manifest_path=str(attempt_manifest_path.relative_to(self.config.project_root)),
            expected_outputs_present=outputs_present,
        )
        manifest = {
            "run_id": run_id,
            "attempt_no": attempt_no,
            "spec": spec.to_dict(),
            "result": result.to_dict(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "git": self._git_state(),
            },
            "input_sha256": input_fingerprints,
            "output_sha256": output_fingerprints,
            "output_fresh": outputs_fresh,
        }
        _atomic_json(attempt_manifest_path, manifest)
        _atomic_json(manifest_path, manifest)
        return result
