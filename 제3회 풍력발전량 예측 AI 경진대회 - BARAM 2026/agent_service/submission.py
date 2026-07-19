from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agent_service.config import ServiceConfig
from agent_service.contracts import RunSpec, resolve_inside
from agent_service.store import AgentStore
from src.metrics import CAPACITY_KWH


KST = ZoneInfo("Asia/Seoul")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateAudit:
    file: str
    file_sha256: str
    size_bytes: int
    rows: int
    columns: tuple[str, ...]
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["columns"] = list(self.columns)
        output["errors"] = list(self.errors)
        return output


class CandidateValidator:
    def __init__(self, config: ServiceConfig):
        self.config = config

    def audit(self, candidate: Path) -> CandidateAudit:
        candidate = Path(candidate).resolve()
        errors: list[str] = []
        if not candidate.is_file():
            return CandidateAudit(str(candidate), "", 0, 0, (), False, ("file does not exist",))
        size = candidate.stat().st_size
        maximum = int(self.config.submission.get("max_file_bytes", 104_857_600))
        if size > maximum:
            errors.append(f"file exceeds {maximum} bytes")
        if candidate.suffix.lower() != ".csv":
            errors.append("candidate must be a CSV file")

        if "competition_profile_path" in self.config.raw:
            profile = self.config.competition_profile
            sample_path = resolve_inside(
                self.config.project_root,
                profile.sample_submission_path,
                "sample_submission_path",
            )
            id_columns = profile.id_columns
            target_ranges = profile.target_ranges
        else:
            sample_path = self.config.project_root / "data" / "sample_submission.csv"
            id_columns = ("forecast_id", "forecast_kst_dtm")
            target_ranges = {
                target: (0.0, capacity) for target, capacity in CAPACITY_KWH.items()
            }
        if not sample_path.is_file():
            errors.append("sample_submission.csv is missing")
            return CandidateAudit(
                str(candidate), sha256_file(candidate), size, 0, (), False, tuple(errors)
            )

        with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample_reader = csv.DictReader(handle)
            sample_rows = list(sample_reader)
            expected_columns = tuple(sample_reader.fieldnames or ())
        try:
            with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = tuple(reader.fieldnames or ())
                rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            errors.append(f"CSV parsing failed: {exc}")
            return CandidateAudit(
                str(candidate), sha256_file(candidate), size, 0, (), False, tuple(errors)
            )
        if columns != expected_columns:
            errors.append("columns do not exactly match sample_submission.csv")
        if len(rows) != len(sample_rows):
            errors.append("row count does not match sample_submission.csv")
        for index, (row, sample) in enumerate(zip(rows, sample_rows)):
            for identifier in id_columns:
                if row.get(identifier) != sample.get(identifier):
                    errors.append(f"{identifier} mismatch at row {index}")
                    break
            if errors:
                break
            for target, bounds in target_ranges.items():
                try:
                    value = float(row[target])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"non-numeric {target} at row {index}")
                    break
                minimum, maximum = bounds
                out_of_range = (
                    not math.isfinite(value)
                    or (minimum is not None and value < minimum)
                    or (maximum is not None and value > maximum)
                )
                if out_of_range:
                    errors.append(f"out-of-range {target} at row {index}")
                    break
            if errors:
                break
        return CandidateAudit(
            file=str(candidate),
            file_sha256=sha256_file(candidate),
            size_bytes=size,
            rows=len(rows),
            columns=columns,
            valid=not errors,
            errors=tuple(errors),
        )


class DaconSubmitter:
    def submit(
        self,
        file_path: Path,
        token: str,
        competition_id: str,
        team_name: str,
        memo: str,
    ) -> dict[str, Any]:
        from dacon_submit_api import dacon_submit_api

        result = dacon_submit_api.post_submission_file(
            str(file_path), token, competition_id, team_name, memo
        )
        if not isinstance(result, dict):
            return {"isSubmitted": False, "detail": "DACON API returned a non-object response"}
        return {
            "isSubmitted": bool(result.get("isSubmitted", False)),
            "detail": str(result.get("detail", "Unknown response")),
        }


class AutomaticSubmissionController:
    def __init__(self, config: ServiceConfig, store: AgentStore):
        self.config = config
        self.store = store
        self.validator = CandidateValidator(config)
        self.submitter = DaconSubmitter()

    def _budget(self) -> dict[str, Any]:
        now_kst = datetime.now(KST)
        start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        end_kst = start_kst + timedelta(days=1)
        return self.store.submission_budget(
            start_kst.astimezone(timezone.utc).isoformat(),
            end_kst.astimezone(timezone.utc).isoformat(),
        )

    def check(self, run_id: int, require_enabled: bool = False) -> dict[str, Any]:
        settings = self.config.submission
        run = self.store.get_run(run_id)
        decision = self.store.get_latest_decision(run_id)
        reasons: list[str] = []
        if decision is None or decision.get("outcome") != "candidate":
            reasons.append("run has not passed the deterministic promotion policy")
        if run["status"] not in {"candidate_pending_human", "auto_submit_ready"}:
            reasons.append(f"run status is not submittable: {run['status']}")
        if self.config.governance.get("require_submission_selection", False):
            selected = self.store.get_active_selection(
                self.config.competition_slug, "submission_candidate"
            )
            if selected is None or int(selected["run_id"]) != run_id:
                reasons.append("run is not the active submission candidate")
        spec = RunSpec.from_dict(run["spec"]).materialize(run_id)
        if not spec.candidate_path:
            reasons.append("run has no candidate path")
            candidate = self.config.submission_dir / "__missing__.csv"
        else:
            candidate = resolve_inside(
                self.config.project_root, spec.candidate_path, "candidate_path"
            )
        audit = self.validator.audit(candidate)
        if not audit.valid:
            reasons.extend(audit.errors)
        if audit.file_sha256 and self.store.has_submission_hash(audit.file_sha256):
            reasons.append("the exact candidate hash was already submitted")

        budget = self._budget()
        if budget["daily"] >= int(settings.get("max_daily_auto_submissions", 1)):
            reasons.append("daily automatic submission budget is exhausted")
        if budget["total"] >= int(settings.get("max_total_auto_submissions", 5)):
            reasons.append("total automatic submission budget is exhausted")
        latest = budget.get("latest_attempted_at")
        if latest:
            last = datetime.fromisoformat(latest)
            minimum = timedelta(hours=float(settings.get("min_hours_between_submissions", 6)))
            if datetime.now(timezone.utc) - last < minimum:
                reasons.append("minimum interval between submissions has not elapsed")
        if require_enabled and not bool(settings.get("auto_submit_enabled", False)):
            reasons.append("auto_submit_enabled is false")

        return {
            "run_id": run_id,
            "eligible": not reasons,
            "reasons": reasons,
            "audit": audit.to_dict(),
            "budget": budget,
            "auto_submit_enabled": bool(settings.get("auto_submit_enabled", False)),
        }

    def submit(self, run_id: int, execute: bool = False) -> dict[str, Any]:
        check = self.check(run_id, require_enabled=execute)
        audit = check["audit"]
        if not check["eligible"]:
            return {**check, "status": "blocked"}
        settings = self.config.submission
        run = self.store.get_run(run_id)
        spec = RunSpec.from_dict(run["spec"]).materialize(run_id)
        candidate = resolve_inside(
            self.config.project_root, str(spec.candidate_path), "candidate_path"
        )
        if not execute:
            response = {"isSubmitted": False, "detail": "dry-run eligible"}
            self.store.record_submission_attempt(
                run_id,
                str(candidate.relative_to(self.config.project_root)),
                audit["file_sha256"],
                True,
                "eligible",
                response,
            )
            return {**check, "status": "dry_run_eligible", "response": response}

        token = os.environ.get(str(settings.get("token_env", "DACON_API_TOKEN")))
        team_name = os.environ.get(str(settings.get("team_name_env", "DACON_TEAM_NAME")))
        if not token or not team_name:
            return {
                **check,
                "status": "blocked",
                "eligible": False,
                "reasons": [*check["reasons"], "DACON token or team name environment variable is missing"],
            }
        memo = f"{settings.get('memo_prefix', 'BARAM agent')} run {run_id}"
        response = self.submitter.submit(
            candidate,
            token,
            str(settings["competition_id"]),
            team_name,
            memo,
        )
        submitted = bool(response.get("isSubmitted"))
        self.store.record_submission_attempt(
            run_id,
            str(candidate.relative_to(self.config.project_root)),
            audit["file_sha256"],
            False,
            "submitted" if submitted else "failed",
            response,
        )
        if submitted:
            self.store.set_run_status(run_id, "submitted_awaiting_score")
            self.store.create_task(
                "leaderboard",
                {
                    "run_id": run_id,
                    "file": candidate.name,
                    "contract": self.config.roles.get("leaderboard", "Record the public score."),
                },
                run_id=run_id,
            )
        return {
            **check,
            "status": "submitted" if submitted else "submission_failed",
            "response": response,
        }
