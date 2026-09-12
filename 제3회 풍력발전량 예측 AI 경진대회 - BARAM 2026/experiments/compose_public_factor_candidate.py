"""Compose one exactly identified public-leaderboard group factor.

The competition metric is the arithmetic mean of three independently evaluated
group metrics.  Consequently, when two scored submissions differ in exactly
one group column, their score/component difference is the exact public effect
of that group vector.  This module transfers such an identified factor onto a
different scored base while failing closed on row, column, or vector mismatch.

This is a leaderboard diagnostic, not fresh out-of-sample model validation.
The report therefore records public-score use explicitly and does not promote
the result as private-leaderboard safe.
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


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
COMPONENTS = ["score", "one_minus_nmae", "ficr"]


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_vector(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float,
) -> bool:
    return bool(
        np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
        )
    )


def compose_identified_factor(
    base: pd.DataFrame,
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    *,
    target: str,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Transfer a single-group treatment whose public effect is identified."""
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    required = ID_COLUMNS + TARGETS
    for name, frame in (
        ("base", base),
        ("treatment", treatment),
        ("control", control),
    ):
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if len(frame) != len(base):
            raise ValueError(f"{name} row count differs from base")
        if not frame[ID_COLUMNS].equals(base[ID_COLUMNS]):
            raise ValueError(f"{name} IDs differ from base")

    if not _same_vector(base[target], control[target], atol=atol):
        raise ValueError(
            "base target vector differs from the scored factor control"
        )
    for other in TARGETS:
        if other == target:
            continue
        if not _same_vector(treatment[other], control[other], atol=atol):
            raise ValueError(
                f"treatment and control differ outside {target}: {other}"
            )

    target_delta = np.abs(
        treatment[target].to_numpy(dtype=float)
        - control[target].to_numpy(dtype=float)
    )
    if not np.any(target_delta > atol):
        raise ValueError("treatment and control have no material target change")

    output = base.copy()
    output[target] = treatment[target].to_numpy(dtype=float)
    return output


def _metrics(
    *,
    score: float,
    one_minus_nmae: float,
    ficr: float,
) -> dict[str, float]:
    values = {
        "score": float(score),
        "one_minus_nmae": float(one_minus_nmae),
        "ficr": float(ficr),
    }
    expected_score = 0.5 * (
        values["one_minus_nmae"] + values["ficr"]
    )
    if not np.isclose(values["score"], expected_score, atol=1e-10, rtol=0.0):
        raise ValueError(
            "score does not equal 0.5 * (one_minus_nmae + ficr)"
        )
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "base": _rooted(args.base),
        "treatment": _rooted(args.treatment),
        "control": _rooted(args.control),
    }
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
    }
    output = compose_identified_factor(
        frames["base"],
        frames["treatment"],
        frames["control"],
        target=args.target,
        atol=args.atol,
    )

    scored = {
        "base": _metrics(
            score=args.base_score,
            one_minus_nmae=args.base_one_minus_nmae,
            ficr=args.base_ficr,
        ),
        "treatment": _metrics(
            score=args.treatment_score,
            one_minus_nmae=args.treatment_one_minus_nmae,
            ficr=args.treatment_ficr,
        ),
        "control": _metrics(
            score=args.control_score,
            one_minus_nmae=args.control_one_minus_nmae,
            ficr=args.control_ficr,
        ),
    }
    identified_delta = {
        component: (
            scored["treatment"][component] - scored["control"][component]
        )
        for component in COMPONENTS
    }
    projected = {
        component: scored["base"][component] + identified_delta[component]
        for component in COMPONENTS
    }
    if not np.isclose(
        projected["score"],
        0.5 * (
            projected["one_minus_nmae"] + projected["ficr"]
        ),
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("projected public components are internally invalid")

    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    control_target = frames["control"][args.target].to_numpy(dtype=float)
    treatment_target = frames["treatment"][args.target].to_numpy(dtype=float)
    movement = np.abs(treatment_target - control_target)
    report = {
        "family": "exact_public_single_group_factor_composition",
        "promotion_tier": "public_confirmed_exploratory",
        "target": args.target,
        "contract": {
            "metric_group_macro_additivity": True,
            "base_target_equals_factor_control": True,
            "factor_pair_differs_only_in_target": True,
            "public_score_used_for_candidate_selection": True,
            "test_actual_generation_used": False,
            "private_leaderboard_transfer_guaranteed": False,
            "numeric_alignment_atol": float(args.atol),
        },
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "scored_public_metrics": scored,
        "identified_public_delta": identified_delta,
        "projected_public_metrics": projected,
        "factor_movement": {
            "changed_rows": int(np.sum(movement > args.atol)),
            "mean_kwh": float(movement.mean()),
            "p95_kwh": float(np.quantile(movement, 0.95)),
            "maximum_kwh": float(movement.max()),
        },
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
        },
        "interpretation": (
            "The projected public value is exact under the published group-macro "
            "metric for the fixed public subset, subject only to DACON scorer "
            "determinism. It is not evidence of equal transfer to the private subset."
        ),
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
    parser.add_argument("--base", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--base-score", type=float, required=True)
    parser.add_argument("--base-one-minus-nmae", type=float, required=True)
    parser.add_argument("--base-ficr", type=float, required=True)
    parser.add_argument("--treatment-score", type=float, required=True)
    parser.add_argument(
        "--treatment-one-minus-nmae",
        type=float,
        required=True,
    )
    parser.add_argument("--treatment-ficr", type=float, required=True)
    parser.add_argument("--control-score", type=float, required=True)
    parser.add_argument(
        "--control-one-minus-nmae",
        type=float,
        required=True,
    )
    parser.add_argument("--control-ficr", type=float, required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "identified_public_delta": report["identified_public_delta"],
                "projected_public_metrics": report["projected_public_metrics"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
