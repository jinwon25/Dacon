from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agent_service.config import ServiceConfig
from agent_service.contracts import RunSpec
from agent_service.orchestrator import Orchestrator


class LeaderboardSynchronizer:
    """Import scored rows from the tracked results CSV and trigger public decisions."""

    def __init__(self, config: ServiceConfig, orchestrator: Orchestrator):
        self.config = config
        self.orchestrator = orchestrator

    def _run_index(self) -> dict[str, dict[str, Any]]:
        output = {}
        for run in self.orchestrator.store.list_rows("runs", limit=10_000):
            candidate = RunSpec.from_dict(run["spec"]).materialize(
                int(run["id"])
            ).candidate_path
            if candidate:
                output[Path(candidate).name] = run
        return output

    def sync(self) -> dict[str, Any]:
        results_path = self.config.project_root / self.config.submission["results_csv"]
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        known = {
            int(row["submission_id"])
            for row in self.orchestrator.store.list_rows("public_results", limit=100_000)
        }
        run_index = self._run_index()
        best_before = float("-inf")
        imported = []
        for row in rows:
            submission_text = str(row.get("submission_id", "")).strip()
            score_text = str(row.get("score", "")).strip()
            if not submission_text or not score_text:
                if score_text:
                    best_before = max(best_before, float(score_text))
                continue
            submission_id = int(submission_text)
            score = float(score_text)
            run = run_index.get(Path(row["file"]).name)
            if submission_id not in known:
                run_id = None if run is None else int(run["id"])
                evaluation = (
                    None
                    if run_id is None
                    else self.orchestrator.store.get_evaluation(run_id)
                )
                family = "legacy_unknown" if evaluation is None else evaluation["family"]
                family_group = (
                    family
                    if evaluation is None
                    else evaluation.get("family_group") or family
                )
                direction = (
                    "unknown"
                    if evaluation is None
                    else str(evaluation.get("direction", "unknown"))
                )
                changed_ratio = None if evaluation is None else evaluation["changed_ratio"]
                delta = 0.0 if best_before == float("-inf") else score - best_before
                payload = {
                    "run_id": run_id,
                    "family": family,
                    "family_group": family_group,
                    "direction": direction,
                    "submission_id": submission_id,
                    "file": row["file"],
                    "score": score,
                    "one_minus_nmae": float(row["one_minus_nmae"]),
                    "ficr": float(row["ficr"]),
                    "score_delta_vs_best": delta,
                    "changed_ratio": changed_ratio,
                    "submitted_at": row["submitted_at"],
                }
                self.orchestrator.record_public_result(payload)
                if run_id is not None and delta < 0.0:
                    self.orchestrator.archive_rejected(run_id, apply=True)
                imported.append(payload)
                known.add(submission_id)
            best_before = max(best_before, score)
        return {"results_path": str(results_path), "imported": imported, "count": len(imported)}
