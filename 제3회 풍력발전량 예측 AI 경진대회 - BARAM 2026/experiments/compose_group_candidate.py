"""Compose independently validated group columns into one submission."""

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
ID_COLUMNS = ["forecast_id", "forecast_kst_dtm"]


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_frames(
    base: pd.DataFrame,
    group1: pd.DataFrame,
    group3: pd.DataFrame,
) -> pd.DataFrame:
    for name, frame in (("group1", group1), ("group3", group3)):
        if not frame[ID_COLUMNS].equals(base[ID_COLUMNS]):
            raise ValueError(f"{name} source IDs differ from base")
    output = base.copy()
    output["kpx_group_1"] = group1["kpx_group_1"].to_numpy(dtype=float)
    output["kpx_group_3"] = group3["kpx_group_3"].to_numpy(dtype=float)
    return output


def _validated_full_delta(
    path: Path,
    target: str,
    *,
    allow_near_stable: bool = False,
    allow_controlled_exploratory: bool = False,
) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    target_validation = report["validation"][target]
    if "promotion_tier" in target_validation:
        tier = target_validation["promotion_tier"]
        stable = tier == "strict" or (
            allow_near_stable and tier == "near_stable"
        )
    elif "promotion" in target_validation:
        stable = target_validation["promotion"] == "promoted"
    elif (
        allow_controlled_exploratory
        and report.get("promotion_tier") == "controlled_exploratory"
    ):
        gates = target_validation.get("gates", {})
        stable = bool(gates) and all(bool(value) for value in gates.values())
    else:
        stable = bool(report["promotion"]["stable"])
    if not stable:
        raise ValueError(f"{target} validation report is not stable")
    return float(target_validation["period_deltas"]["full"]["score"])


def _stable_full_delta(path: Path, target: str) -> float:
    return _validated_full_delta(path, target)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "base": _rooted(args.base),
        "group1": _rooted(args.group1_source),
        "group3": _rooted(args.group3_source),
        "group1_validation": _rooted(args.group1_validation),
        "group3_validation": _rooted(args.group3_validation),
    }
    if args.group2_validation:
        paths["group2_validation"] = _rooted(args.group2_validation)
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
        if name in {"base", "group1", "group3"}
    }
    output = compose_frames(
        frames["base"],
        frames["group1"],
        frames["group3"],
    )
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    full_deltas = {
        "kpx_group_1": _validated_full_delta(
            paths["group1_validation"],
            "kpx_group_1",
            allow_controlled_exploratory=bool(
                args.allow_controlled_exploratory_group1
            ),
        ),
        "kpx_group_3": _stable_full_delta(
            paths["group3_validation"],
            "kpx_group_3",
        ),
    }
    if args.group2_validation:
        full_deltas["kpx_group_2"] = _validated_full_delta(
            paths["group2_validation"],
            "kpx_group_2",
            allow_near_stable=bool(args.allow_near_stable_group2),
        )
    expected_macro_delta = float(sum(full_deltas.values()) / 3.0)
    movement: dict[str, Any] = {}
    for target, capacity in CAPACITY_KWH.items():
        delta = np.abs(
            output[target].to_numpy(dtype=float)
            - frames["base"][target].to_numpy(dtype=float)
        )
        movement[target] = {
            "changed_rows": int(np.sum(delta > 1e-6)),
            "mean_kwh": float(delta.mean()),
            "p95_kwh": float(np.quantile(delta, 0.95)),
            "maximum_kwh": float(delta.max()),
            "maximum_capacity_ratio": float(delta.max() / capacity),
        }
    report = {
        "family": "independently_validated_group_column_composition",
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "contract": {
            "kpx_group_1": args.group1_contract,
            "kpx_group_2": args.group2_contract,
            "kpx_group_3": args.group3_contract,
            "public_score_used_for_model_selection": False,
            "test_actual_generation_used": False,
            "controlled_exploratory_group1_enabled": bool(
                args.allow_controlled_exploratory_group1
            ),
        },
        "local_full_score_deltas": full_deltas,
        "expected_macro_score_delta_if_2024_transfers": expected_macro_delta,
        "projected_public_score_if_local_delta_transfers": float(
            args.base_public_score + expected_macro_delta
        ),
        "movement_vs_base": movement,
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
        },
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument("--group1-source", required=True)
    parser.add_argument("--group3-source", required=True)
    parser.add_argument("--group1-validation", required=True)
    parser.add_argument("--group3-validation", required=True)
    parser.add_argument("--group2-validation")
    parser.add_argument(
        "--group1-contract",
        default="stable UMRG year-forward expert",
    )
    parser.add_argument(
        "--group3-contract",
        default=(
            "stable actual-UMKR-train to locally-emulated-UMKR-query "
            "year-forward expert"
        ),
    )
    parser.add_argument(
        "--group2-contract",
        default="unchanged public-confirmed alpha-0.2375 incumbent",
    )
    parser.add_argument(
        "--allow-near-stable-group2",
        action="store_true",
    )
    parser.add_argument(
        "--allow-controlled-exploratory-group1",
        action="store_true",
    )
    parser.add_argument("--base-public-score", type=float, default=0.6440998116)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "local_full_score_deltas": report["local_full_score_deltas"],
                "expected_macro_score_delta": report[
                    "expected_macro_score_delta_if_2024_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_local_delta_transfers"
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
