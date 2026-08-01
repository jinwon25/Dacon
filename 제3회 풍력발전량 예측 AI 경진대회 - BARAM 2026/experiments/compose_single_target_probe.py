"""Compose one explicitly risk-accepted target probe onto an incumbent.

The utility is deliberately narrow: it copies exactly one target column from
a production source, proves every other column remains byte-for-byte equal to
the incumbent frame, and records the failed-family evidence that required a
human risk override.  It prepares a CSV but never submits it externally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")
TARGETS = tuple(CAPACITY_KWH)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_single_target(
    incumbent: pd.DataFrame,
    source: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    """Copy only ``target`` while preserving the exact incumbent schema."""
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    if list(source.columns) != list(incumbent.columns):
        raise ValueError("source columns differ from incumbent")
    if len(source) != len(incumbent):
        raise ValueError("source row count differs from incumbent")
    if not source[list(ID_COLUMNS)].equals(incumbent[list(ID_COLUMNS)]):
        raise ValueError("source identifiers differ from incumbent")

    output = incumbent.copy()
    output[target] = source[target].to_numpy(dtype=float)
    for column in incumbent.columns:
        if column == target:
            continue
        if not output[column].equals(incumbent[column]):
            raise AssertionError(f"unexpected non-target change: {column}")
    if np.array_equal(
        output[target].to_numpy(dtype=float),
        incumbent[target].to_numpy(dtype=float),
    ):
        raise ValueError("source target is identical to incumbent")
    return output


def _risk_evidence(path: Path, target: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        record = report["active_replacements"]["multiyear"]["variants"][
            target
        ]
    except KeyError as exc:
        raise ValueError("risk report lacks multiyear target evidence") from exc
    if not bool(record.get("statistical_gate_eligible")):
        raise ValueError("target did not pass the statistical gate")
    if bool(record.get("promotion_eligible")):
        raise ValueError("risk override is unnecessary for an eligible target")
    if record.get("decision") != "rejected_historical_public_failure_guard":
        raise ValueError("target was not rejected solely by the public-failure guard")
    return {
        "statistical_gate_eligible": True,
        "promotion_eligible": False,
        "decision": record["decision"],
        "full_period_delta": record["period_deltas"]["full"],
        "movement": record["movement"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.accept_public_failure_risk:
        raise ValueError(
            "probe creation requires --accept-public-failure-risk"
        )
    paths = {
        "incumbent": _rooted(args.incumbent),
        "source": _rooted(args.source),
        "risk_report": _rooted(args.risk_report),
    }
    evidence = _risk_evidence(paths["risk_report"], args.target)
    incumbent = pd.read_csv(paths["incumbent"], encoding="utf-8-sig")
    source = pd.read_csv(paths["source"], encoding="utf-8-sig")
    output = compose_single_target(incumbent, source, target=args.target)

    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    incumbent_values = incumbent[args.target].to_numpy(dtype=float)
    output_values = output[args.target].to_numpy(dtype=float)
    movement = np.abs(output_values - incumbent_values)
    unchanged = {
        target: bool(output[target].equals(incumbent[target]))
        for target in TARGETS
        if target != args.target
    }
    report = {
        "family": "single_target_risk_accepted_probe",
        "target": args.target,
        "contract": {
            "risk_accepted_by_user": True,
            "historical_public_failure_guard_overridden": True,
            "statistical_gate_required": True,
            "non_target_columns_unchanged": all(unchanged.values()),
            "external_submission_executed": False,
        },
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "risk_evidence": evidence,
        "non_target_equality": unchanged,
        "target_movement": {
            "changed_rows": int(np.sum(movement > 1e-8)),
            "mean_kwh": float(np.mean(movement)),
            "p95_kwh": float(np.quantile(movement, 0.95)),
            "maximum_kwh": float(np.max(movement)),
            "maximum_capacity_ratio": float(
                np.max(movement) / CAPACITY_KWH[args.target]
            ),
        },
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": len(output),
            "audit": audit.to_dict(),
        },
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--risk-report", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accept-public-failure-risk", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "risk_evidence": report["risk_evidence"],
                "target_movement": report["target_movement"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
