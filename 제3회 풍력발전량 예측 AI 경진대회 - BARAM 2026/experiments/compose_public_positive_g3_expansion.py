"""Expand the already public-positive group-3 KMA/GFS factor.

The group-3 control and treatment were submitted in submissions 27 and 31.
Only group 3 changed, so their public metric difference is an isolated factor
observation.  This module applies a bounded 1.25x dose of that same factor on
top of the selected group-1/group-2 candidate.  It does not fit on 2025 data.
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
from experiments.compose_public_positive_multifactor_candidate import (
    COMPONENTS,
    _single_target_delta,
)
from experiments.kma_um_power_curve_gate import issue_block_bootstrap
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")
TARGET = "kpx_group_3"
PUBLIC_G3_FACTOR_DELTA = {
    "score": 0.0004896767,
    "one_minus_nmae": -0.0002512213,
    "ficr": 0.0012305746,
}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_group3_factor(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    multiplier: float,
    capacity: float,
    maximum_factor_ratio: float,
) -> np.ndarray:
    """Scale one observed factor while preserving physical bounds."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.shape != treatment.shape:
        raise ValueError("group-3 factor vectors do not align")
    if multiplier < 1.0:
        raise ValueError("multiplier must preserve at least the scored treatment")
    if not 0.0 < maximum_factor_ratio <= 1.0:
        raise ValueError("maximum factor ratio must lie in (0, 1]")
    movement = np.clip(
        multiplier * (treatment - control),
        -maximum_factor_ratio * capacity,
        maximum_factor_ratio * capacity,
    )
    return np.clip(control + movement, 0.0, capacity)


def _movement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    absolute = np.abs(delta)
    return {
        "changed_rows": int((absolute > 1e-9).sum()),
        "changed_fraction": float((absolute > 1e-9).mean()),
        "mean_absolute_kwh": float(absolute.mean()),
        "p95_absolute_kwh": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_kwh": float(absolute.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "anchor": _rooted(args.anchor),
        "anchor_report": _rooted(args.anchor_report),
        "group3_control": _rooted(args.group3_control),
        "group3_treatment": _rooted(args.group3_treatment),
        "group3_oof": _rooted(args.group3_oof),
    }
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
        if name in ("anchor", "group3_control", "group3_treatment")
    }
    anchor = frames["anchor"]
    for name, frame in frames.items():
        if not anchor[list(ID_COLUMNS)].equals(frame[list(ID_COLUMNS)]):
            raise ValueError(f"{name} identifiers differ from anchor")
    anchor_g3 = anchor[TARGET].to_numpy(dtype=float)
    treatment_g3 = frames["group3_treatment"][TARGET].to_numpy(dtype=float)
    if not np.allclose(anchor_g3, treatment_g3, rtol=0.0, atol=1e-8):
        raise ValueError("anchor does not contain the scored group-3 treatment")

    cache = np.load(paths["group3_oof"])
    index = pd.to_datetime(cache["index_ns"])
    truth = cache["truth"].astype(float)
    oof_control = cache["public_fine"].astype(float)
    oof_treatment = cache["rolling_candidate"].astype(float)
    available = cache["available"].astype(bool)
    q2 = cache["q2"].astype(bool)
    h2 = cache["h2"].astype(bool)
    q1 = available & np.asarray(index < pd.Timestamp("2024-04-01"))
    full = available & np.asarray(index >= pd.Timestamp("2024-01-01"))
    capacity = float(CAPACITY_KWH[TARGET])
    oof_candidate = expand_group3_factor(
        oof_control,
        oof_treatment,
        multiplier=args.multiplier,
        capacity=capacity,
        maximum_factor_ratio=args.maximum_factor_ratio,
    )

    masks = {"q1": q1, "q2": q2, "h2": h2, "full": full}
    period_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in masks.items():
        period_deltas[name] = {
            "total_vs_control": _single_target_delta(
                truth, oof_control, oof_candidate, rows, TARGET
            ),
            "incremental_vs_treatment": _single_target_delta(
                truth, oof_treatment, oof_candidate, rows, TARGET
            ),
        }
    monthly_incremental = {}
    for month in range(7, 13):
        rows = h2 & np.asarray(index.month == month)
        monthly_incremental[str(month)] = _single_target_delta(
            truth, oof_treatment, oof_candidate, rows, TARGET
        )
    bootstrap = issue_block_bootstrap(
        truth,
        oof_treatment,
        oof_candidate,
        cache["issue_ns"],
        h2,
        n_bootstrap=args.bootstrap_repetitions,
        seed=args.seed,
    )

    base_local = _single_target_delta(
        truth, oof_control, oof_treatment, full, TARGET
    )
    total_local = period_deltas["full"]["total_vs_control"]
    projected_total = {
        component: float(
            PUBLIC_G3_FACTOR_DELTA[component]
            * total_local[component]
            / base_local[component]
        )
        for component in COMPONENTS
    }
    projected_increment = {
        component: float(
            projected_total[component] - PUBLIC_G3_FACTOR_DELTA[component]
        )
        for component in COMPONENTS
    }
    anchor_report = json.loads(paths["anchor_report"].read_text(encoding="utf-8"))
    anchor_projection = anchor_report["submission_1508365_calibrated_projection"][
        "projected_public_metrics"
    ]
    combined_projection = {
        component: float(anchor_projection[component] + projected_increment[component])
        for component in COMPONENTS
    }

    gates = {
        "full_incremental_score_positive": period_deltas["full"][
            "incremental_vs_treatment"
        ]["score"]
        > 0.0,
        "full_incremental_ficr_positive": period_deltas["full"][
            "incremental_vs_treatment"
        ]["ficr"]
        > 0.0,
        "h2_incremental_all_components_positive": min(
            period_deltas["h2"]["incremental_vs_treatment"].values()
        )
        > 0.0,
        "all_h2_month_scores_positive": all(
            row["score"] > 0.0 for row in monthly_incremental.values()
        ),
        "bootstrap_q05_all_components_positive": min(bootstrap["q05"].values())
        > 0.0,
        "bootstrap_positive_fraction_pass": bootstrap[
            "positive_all_component_fraction"
        ]
        >= args.minimum_bootstrap_positive_fraction,
        "projected_score_improves_anchor": combined_projection["score"]
        > anchor_projection["score"],
    }
    qualified = bool(all(gates.values()))
    if not qualified:
        raise RuntimeError(f"group-3 expansion failed closed: {gates}")

    production_candidate_g3 = expand_group3_factor(
        frames["group3_control"][TARGET].to_numpy(dtype=float),
        treatment_g3,
        multiplier=args.multiplier,
        capacity=capacity,
        maximum_factor_ratio=args.maximum_factor_ratio,
    )
    output = anchor.copy()
    output[TARGET] = production_candidate_g3
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    report = {
        "schema_version": "public_positive_group3_expansion.v1",
        "family": "public_positive_g1_g2_anchor_plus_group3_dose",
        "contract": {
            "group3_multiplier": args.multiplier,
            "maximum_factor_ratio": args.maximum_factor_ratio,
            "public_isolated_factor_used": True,
            "test_actual_generation_used": False,
        },
        "sources": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "validation": {
            "period_deltas": period_deltas,
            "h2_monthly_incremental": monthly_incremental,
            "h2_issue_block_bootstrap": bootstrap,
            "oof_movement_vs_treatment": _movement(
                oof_treatment, oof_candidate
            ),
            "promotion_gates": gates,
        },
        "public_projection": {
            "observed_group3_factor_delta": PUBLIC_G3_FACTOR_DELTA,
            "local_base_factor_delta": base_local,
            "local_total_factor_delta": total_local,
            "projected_group3_total": projected_total,
            "projected_group3_increment": projected_increment,
            "anchor_metrics": anchor_projection,
            "combined_metrics": combined_projection,
            "score_gap_to_065": float(0.65 - combined_projection["score"]),
            "warning": (
                "The group-3 response extrapolates one isolated public factor point "
                "with its OOF dose-response ratio; it is not an observed score."
            ),
        },
        "production_movement_vs_anchor": _movement(
            anchor_g3, production_candidate_g3
        ),
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
        },
        "decision": "controlled_public_probe_candidate",
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
        "--anchor",
        default=(
            "artifacts_final/candidates/"
            "public_positive_g1w1375_g2w1825_g3frozen_20260802.csv"
        ),
    )
    parser.add_argument(
        "--anchor-report",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_g1w1375_g2w1825_g3frozen_20260802.json"
        ),
    )
    parser.add_argument(
        "--group3-control",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--group3-treatment",
        default="submissions/blend_best_kma_um_power_curve_gate.csv",
    )
    parser.add_argument(
        "--group3-oof",
        default=(
            "artifacts_final/external_weather/kma_um_regional_context_2024/"
            "power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument("--multiplier", type=float, default=1.25)
    parser.add_argument("--maximum-factor-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-bootstrap-positive-fraction", type=float, default=0.95)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "public_positive_g1w1375_g2w1825_g3x125_20260802.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_g1w1375_g2w1825_g3x125_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "candidate": report["candidate"],
                "promotion_gates": report["validation"]["promotion_gates"],
                "public_projection": report["public_projection"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
