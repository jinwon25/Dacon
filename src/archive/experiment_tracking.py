"""Append-only experiment logging."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

FIELDS = [
    "experiment_id",
    "run_date",
    "validation_split",
    "feature_groups",
    "model_params",
    "seed",
    "raw_brier_score",
    "calibrated_brier_score",
    "local_brier_skill_score",
    "calibration",
    "training_seconds",
    "inference_seconds",
    "model_size_mb",
    "peak_memory_mb",
    "leakage_risk_notes",
]


def append_experiment(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    row = {field: record.get(field, "") for field in FIELDS}
    row["run_date"] = row["run_date"] or datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
