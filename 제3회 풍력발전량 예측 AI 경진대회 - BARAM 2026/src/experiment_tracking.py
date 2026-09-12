from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "experiment_id", "hypothesis", "feature_set", "model", "hyperparameters",
    "fold_scores", "group_metrics", "mean_delta", "worst_fold_delta",
    "bootstrap_interval", "runtime_seconds", "decision", "reason",
]


def save_oof_predictions(frame: pd.DataFrame, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {"timestamp", "target", "fold", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")
    identity = ["timestamp", "target", "fold"]
    for optional in ("candidate", "policy"):
        if optional in frame:
            identity.append(optional)
    if frame.duplicated(identity).any():
        raise ValueError(f"OOF predictions contain duplicate rows for identity {identity}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def append_experiment_result(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(RESULT_COLUMNS) - set(result)
    if missing:
        raise ValueError(f"Missing experiment result fields: {sorted(missing)}")
    row = {key: json.dumps(result[key], ensure_ascii=False) if isinstance(result[key], (dict, list, tuple)) else result[key] for key in RESULT_COLUMNS}
    if path.exists():
        current = pd.read_csv(path, encoding="utf-8-sig")
        current = current.loc[current["experiment_id"] != row["experiment_id"]]
        output = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    else:
        output = pd.DataFrame([row], columns=RESULT_COLUMNS)
    output.to_csv(path, index=False, encoding="utf-8-sig")
