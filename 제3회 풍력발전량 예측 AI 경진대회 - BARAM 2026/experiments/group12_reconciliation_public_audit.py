"""Audit local-to-public transfer of the scored G1/G2 reconciliation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_submission(path: Path, submission_id: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["submission_id"] == submission_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one result for submission {submission_id}, got {len(matches)}"
        )
    return {component: float(matches[0][component]) for component in COMPONENTS}


def transfer_breakdown(
    local_pair_delta: dict[str, float],
    public_macro_delta: dict[str, float],
) -> dict[str, Any]:
    """Compare a two-group local delta with a three-group public macro delta."""
    public_pair_delta = {
        component: 1.5 * float(public_macro_delta[component])
        for component in COMPONENTS
    }
    local_macro_equivalent = {
        component: (2.0 / 3.0) * float(local_pair_delta[component])
        for component in COMPONENTS
    }
    ratios = {
        component: (
            public_pair_delta[component] / float(local_pair_delta[component])
            if abs(float(local_pair_delta[component])) > 1e-15
            else None
        )
        for component in COMPONENTS
    }
    return {
        "local_pair_delta": local_pair_delta,
        "local_macro_equivalent": local_macro_equivalent,
        "public_macro_delta": public_macro_delta,
        "public_pair_delta": public_pair_delta,
        "pair_transfer_ratio": ratios,
        "sign_agreement": {
            component: bool(
                local_pair_delta[component] * public_pair_delta[component] > 0.0
            )
            for component in COMPONENTS
        },
        "near_mirror_score_reversal": bool(
            abs(
                public_pair_delta["score"] + local_pair_delta["score"]
            )
            <= 0.05 * abs(local_pair_delta["score"])
        ),
    }


def run(
    results_path: Path,
    local_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    incumbent = _load_submission(results_path, "1502437")
    treatment = _load_submission(results_path, "1504383")
    public_macro_delta = {
        component: treatment[component] - incumbent[component]
        for component in COMPONENTS
    }
    local = json.loads(local_report_path.read_text(encoding="utf-8"))
    local_pair_delta = local["validation"]["period_deltas"]["full"]
    breakdown = transfer_breakdown(local_pair_delta, public_macro_delta)
    report = {
        "family": "group12_zero_sum_difference_reconciliation",
        "submissions": {
            "control": "1502437",
            "treatment": "1504383",
        },
        **breakdown,
        "diagnosis": {
            "mean_error_signal_transferred": (
                breakdown["sign_agreement"]["one_minus_nmae"]
            ),
            "settlement_signal_reversed": (
                not breakdown["sign_agreement"]["ficr"]
            ),
            "broad_postprocessing_closed": True,
            "next_method": (
                "sparse high-confidence movement whose historical absolute "
                "error is predicted to improve, so settlement cannot worsen "
                "on correctly selected rows"
            ),
            "public_result_used_for_method_family_prioritization_only": True,
            "public_result_must_not_select_gate_or_threshold": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="submissions/results.csv")
    parser.add_argument(
        "--local-report",
        default=(
            "artifacts_final/diagnostics/"
            "group12_difference_reconciliation_unanimous_w10_20260728.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "group12_reconciliation_public_result_20260728.json"
        ),
    )
    args = parser.parse_args()
    report = run(
        _rooted(args.results),
        _rooted(args.local_report),
        _rooted(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
