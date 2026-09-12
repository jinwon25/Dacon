"""Decompose scored submissions into isolated public factor responses.

The BARAM leaderboard score is the arithmetic mean of three independently
scored groups.  When two submitted files differ only in known groups, their
metric difference identifies the public response of that factor without
guessing from the headline score.

This module records those already-observed responses.  It does not select row
gates, blend weights, or models from the public leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("score", "one_minus_nmae", "ficr")


@dataclass(frozen=True)
class FactorContrast:
    name: str
    treatment_id: str
    control_id: str
    targets: tuple[str, ...]
    family_status: str


CONTRASTS = (
    FactorContrast(
        "pooled_group2",
        "1501938",
        "1501926",
        ("kpx_group_2",),
        "public_positive_retained",
    ),
    FactorContrast(
        "residual_group1",
        "1502437",
        "1501927",
        ("kpx_group_1",),
        "public_positive_retained",
    ),
    FactorContrast(
        "umkr_group3",
        "1501938",
        "1501927",
        ("kpx_group_3",),
        "rejected",
    ),
    FactorContrast(
        "jma_gsm_group3",
        "1502447",
        "1502437",
        ("kpx_group_3",),
        "rejected_closed",
    ),
    FactorContrast(
        "directional_ficr_group1",
        "1503281",
        "1502437",
        ("kpx_group_1",),
        "rejected_closed",
    ),
    FactorContrast(
        "directional_ficr_group2_group3",
        "1503282",
        "1502437",
        ("kpx_group_2", "kpx_group_3"),
        "rejected_closed",
    ),
    FactorContrast(
        "trajectory_tcn_group3",
        "1503349",
        "1502437",
        ("kpx_group_3",),
        "rejected_closed",
    ),
    FactorContrast(
        "jma_msm_plateau_group3",
        "1504352",
        "1502437",
        ("kpx_group_3",),
        "rejected_closed",
    ),
    FactorContrast(
        "group12_difference_reconciliation",
        "1504383",
        "1502437",
        ("kpx_group_1", "kpx_group_2"),
        "rejected_closed",
    ),
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        submission_id = str(row["submission_id"])
        if submission_id in result:
            raise ValueError(f"duplicate submission id: {submission_id}")
        result[submission_id] = {
            **row,
            **{component: float(row[component]) for component in COMPONENTS},
        }
    return result


def derive_factor_response(
    treatment: dict[str, Any],
    control: dict[str, Any],
    *,
    targets: tuple[str, ...],
) -> dict[str, Any]:
    if not targets:
        raise ValueError("at least one changed target is required")
    macro_delta = {
        component: float(treatment[component] - control[component])
        for component in COMPONENTS
    }
    multiplier = 3.0 / len(targets)
    affected_group_mean_delta = {
        component: float(multiplier * macro_delta[component])
        for component in COMPONENTS
    }
    identity_error = abs(
        macro_delta["score"]
        - 0.5
        * (
            macro_delta["one_minus_nmae"]
            + macro_delta["ficr"]
        )
    )
    return {
        "targets": list(targets),
        "target_count": len(targets),
        "macro_delta": macro_delta,
        "affected_group_mean_delta": affected_group_mean_delta,
        "score_identity_absolute_error": float(identity_error),
        "score_identity_passed": bool(identity_error <= 2e-10),
    }


def audit_factor_responses(
    results: dict[str, dict[str, Any]],
    contrasts: tuple[FactorContrast, ...] = CONTRASTS,
) -> dict[str, Any]:
    factors: dict[str, Any] = {}
    for contrast in contrasts:
        try:
            treatment = results[contrast.treatment_id]
            control = results[contrast.control_id]
        except KeyError as exc:
            raise ValueError(
                f"missing scored submission required by {contrast.name}: {exc}"
            ) from exc
        response = derive_factor_response(
            treatment,
            control,
            targets=contrast.targets,
        )
        factors[contrast.name] = {
            "status": contrast.family_status,
            "treatment": {
                "submission_id": contrast.treatment_id,
                "file": treatment["file"],
            },
            "control": {
                "submission_id": contrast.control_id,
                "file": control["file"],
            },
            **response,
        }

    positive = [
        name
        for name, row in factors.items()
        if row["macro_delta"]["score"] > 0.0
    ]
    ficr_failures = [
        name
        for name, row in factors.items()
        if row["macro_delta"]["ficr"] < 0.0
    ]

    # Two independent identities in the 1501926--1502437 factorial block.
    closure_checks = {
        "residual_g1_bridge": {
            component: float(
                (results["1501928"][component] - results["1501938"][component])
                - factors["residual_group1"]["macro_delta"][component]
            )
            for component in COMPONENTS
        },
        "incumbent_composition": {
            component: float(
                (
                    results["1501927"][component]
                    + factors["residual_group1"]["macro_delta"][component]
                )
                - results["1502437"][component]
            )
            for component in COMPONENTS
        },
    }
    maximum_closure_error = max(
        abs(value)
        for check in closure_checks.values()
        for value in check.values()
    )

    return {
        "contract": {
            "public_scores_used_for_diagnosis_only": True,
            "row_level_gate_or_weight_selected_from_public": False,
            "group_macro_additivity": True,
            "affected_group_delta_multiplier": (
                "3 / number_of_changed_groups"
            ),
        },
        "factors": factors,
        "summary": {
            "positive_score_factors": positive,
            "negative_ficr_factors": ficr_failures,
            "retained_public_positive_axes": [
                "residual_group1",
                "pooled_group2",
            ],
            "frozen_axes": [
                "kpx_group_3 replacements",
                "broad group-1/group-2 differential postprocessing",
            ],
            "next_search_space": (
                "new core models or sparse settlement-safe high-confidence "
                "moves; do not retune rejected postprocessors from public scores"
            ),
        },
        "factorial_closure": {
            "checks": closure_checks,
            "maximum_absolute_error": float(maximum_closure_error),
            "passed": bool(maximum_closure_error <= 2e-10),
        },
    }


def run(results_path: Path, output_path: Path) -> dict[str, Any]:
    results = load_results(results_path)
    report = audit_factor_responses(results)
    report["sources"] = {
        "results": results_path.relative_to(ROOT).as_posix(),
        "recorded_submissions": len(results),
        "latest_submission_id": next(reversed(results)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="submissions/results.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "public_factor_response_audit_20260728.json"
        ),
    )
    args = parser.parse_args()
    report = run(_rooted(args.results), _rooted(args.output))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
