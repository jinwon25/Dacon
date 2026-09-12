"""Compose a capped, fixed residual stack from validated group experts.

The primary group-1 candidate is augmented by a fixed fraction of a second
candidate's movement from the same incumbent.  The combined movement remains
inside the original capacity-relative cap.  Group 2 and group 3 are copied
from independently validated source candidates.

This experiment is deliberately reported as controlled exploratory because
the residual fraction was screened on the contaminated 2024 historical
benchmark.  It is never an automatic strict promotion.
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
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.compose_group_candidate import _validated_full_delta
from experiments.kma_year_forward_quantile_blend import _periods, metric_delta
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
TARGET = "kpx_group_1"


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_capped_residual_stack(
    reference: np.ndarray,
    primary: np.ndarray,
    residual_source: np.ndarray,
    *,
    residual_weight: float,
    capacity: float,
    movement_cap_ratio: float,
) -> np.ndarray:
    """Add a fixed residual-source movement and preserve the total cap."""
    reference = np.asarray(reference, dtype=float)
    primary = np.asarray(primary, dtype=float)
    residual_source = np.asarray(residual_source, dtype=float)
    if not (
        reference.shape == primary.shape == residual_source.shape
    ):
        raise ValueError("residual-stack vectors do not align")
    if not 0.0 <= residual_weight <= 1.0:
        raise ValueError("residual weight must lie in [0, 1]")
    if not 0.0 < movement_cap_ratio <= 1.0:
        raise ValueError("movement cap ratio must lie in (0, 1]")
    raw = primary + residual_weight * (residual_source - reference)
    lower = np.clip(
        reference - movement_cap_ratio * capacity,
        0.0,
        capacity,
    )
    upper = np.clip(
        reference + movement_cap_ratio * capacity,
        0.0,
        capacity,
    )
    return np.clip(raw, lower, upper)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _require_same_ids(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    name: str,
) -> None:
    if not reference[ID_COLUMNS].equals(other[ID_COLUMNS]):
        raise ValueError(f"{name} IDs differ from the incumbent")


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "incumbent": _rooted(args.incumbent),
        "primary_candidate": _rooted(args.primary_candidate),
        "residual_candidate": _rooted(args.residual_candidate),
        "group3_candidate": _rooted(args.group3_candidate),
        "primary_cache": _rooted(args.primary_cache),
        "residual_cache": _rooted(args.residual_cache),
        "primary_report": _rooted(args.primary_report),
        "residual_report": _rooted(args.residual_report),
        "group2_report": _rooted(args.group2_report),
        "group3_report": _rooted(args.group3_report),
    }
    frames = {
        name: _load_frame(paths[name])
        for name in (
            "incumbent",
            "primary_candidate",
            "residual_candidate",
            "group3_candidate",
        )
    }
    for name, frame in frames.items():
        if name != "incumbent":
            _require_same_ids(frames["incumbent"], frame, name)

    primary_cache = np.load(paths["primary_cache"], allow_pickle=False)
    residual_cache = np.load(paths["residual_cache"], allow_pickle=False)
    prefix = f"{TARGET}__"
    if not np.array_equal(
        primary_cache[f"{prefix}index_ns"],
        residual_cache[f"{prefix}index_ns"],
    ):
        raise ValueError("validation cache indexes differ")
    for key in ("truth", "reference"):
        if not np.allclose(
            primary_cache[f"{prefix}{key}"],
            residual_cache[f"{prefix}{key}"],
        ):
            raise ValueError(f"validation cache {key} vectors differ")

    capacity = CAPACITY_KWH[TARGET]
    index = pd.DatetimeIndex(
        pd.to_datetime(primary_cache[f"{prefix}index_ns"])
    )
    truth = primary_cache[f"{prefix}truth"].astype(float)
    reference = primary_cache[f"{prefix}reference"].astype(float)
    validation_candidate = apply_capped_residual_stack(
        reference,
        primary_cache[f"{prefix}candidate"].astype(float),
        residual_cache[f"{prefix}candidate"].astype(float),
        residual_weight=args.residual_weight,
        capacity=capacity,
        movement_cap_ratio=args.movement_cap_ratio,
    )
    periods = _periods(index)
    period_deltas = {
        name: metric_delta(
            truth,
            reference,
            validation_candidate,
            capacity,
            rows,
        )
        for name, rows in periods.items()
    }
    monthly = {
        str(month): metric_delta(
            truth,
            reference,
            validation_candidate,
            capacity,
            periods["full"] & np.asarray(index.month == month),
        )
        for month in range(1, 13)
    }
    positive_months = int(
        sum(value["score"] > 0.0 for value in monthly.values())
    )
    issue = pd.DatetimeIndex(
        pd.to_datetime(primary_cache[f"{prefix}issue_ns"])
    )
    bootstrap = evaluate_blocked_rolling(
        truth,
        reference,
        validation_candidate,
        index,
        issue,
        periods["full"] & (truth >= 0.10 * capacity),
        n_bootstrap=args.n_bootstrap,
        seed=20260726,
    )
    gates = {
        "q1_components_positive": min(period_deltas["q1"].values()) > 0.0,
        "q2_components_positive": min(period_deltas["q2"].values()) > 0.0,
        "h2_components_positive": min(period_deltas["h2"].values()) > 0.0,
        "full_components_positive": min(period_deltas["full"].values()) > 0.0,
        "positive_score_month_fraction_at_least_80pct": (
            positive_months / 12.0 >= 0.80
        ),
        "issue_bootstrap_q05_positive": (
            bootstrap["issue_block_bootstrap"]["q05"] > 0.0
        ),
        "issue_bootstrap_positive_fraction_at_least_98pct": (
            bootstrap["issue_block_bootstrap"]["positive_fraction"] >= 0.98
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"residual stack failed validation gates: {gates}")

    production_group1 = apply_capped_residual_stack(
        frames["incumbent"][TARGET].to_numpy(dtype=float),
        frames["primary_candidate"][TARGET].to_numpy(dtype=float),
        frames["residual_candidate"][TARGET].to_numpy(dtype=float),
        residual_weight=args.residual_weight,
        capacity=capacity,
        movement_cap_ratio=args.movement_cap_ratio,
    )
    output = frames["primary_candidate"].copy()
    output[TARGET] = production_group1
    output["kpx_group_3"] = frames["group3_candidate"][
        "kpx_group_3"
    ].to_numpy(dtype=float)
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    group_deltas = {
        TARGET: float(period_deltas["full"]["score"]),
        "kpx_group_2": _validated_full_delta(
            paths["group2_report"],
            "kpx_group_2",
            allow_near_stable=True,
        ),
        "kpx_group_3": _validated_full_delta(
            paths["group3_report"],
            "kpx_group_3",
        ),
    }
    # Ensure both group-1 source experts were independently stable before
    # using the post-hoc residual composition.
    _validated_full_delta(paths["primary_report"], TARGET)
    _validated_full_delta(paths["residual_report"], TARGET)
    expected_macro_delta = float(sum(group_deltas.values()) / 3.0)
    movement = np.abs(
        production_group1
        - frames["incumbent"][TARGET].to_numpy(dtype=float)
    )
    report = {
        "family": "fixed_capped_residual_stack",
        "promotion_tier": "controlled_exploratory",
        "selection_caveat": (
            "the 1/8 residual fraction was screened on the repeatedly inspected "
            "2024 historical benchmark; public score was not used"
        ),
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "contract": {
            "residual_weight": float(args.residual_weight),
            "movement_cap_ratio": float(args.movement_cap_ratio),
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "group2_tier": "near_stable",
            "group3_tier": "strict",
        },
        "validation": {
            TARGET: {
                "period_deltas": period_deltas,
                "monthly_deltas": monthly,
                "positive_score_months": positive_months,
                "issue_block_validation": bootstrap,
                "gates": gates,
            }
        },
        "local_full_score_deltas": group_deltas,
        "expected_macro_score_delta_if_2024_transfers": expected_macro_delta,
        "projected_public_score_if_local_delta_transfers": float(
            args.base_public_score + expected_macro_delta
        ),
        "movement": {
            TARGET: {
                "changed_rows": int(np.sum(movement > 1e-6)),
                "mean_kwh": float(movement.mean()),
                "p95_kwh": float(np.quantile(movement, 0.95)),
                "maximum_kwh": float(movement.max()),
                "maximum_capacity_ratio": float(movement.max() / capacity),
            }
        },
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
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument("--residual-candidate", required=True)
    parser.add_argument("--group3-candidate", required=True)
    parser.add_argument("--primary-cache", required=True)
    parser.add_argument("--residual-cache", required=True)
    parser.add_argument("--primary-report", required=True)
    parser.add_argument("--residual-report", required=True)
    parser.add_argument("--group2-report", required=True)
    parser.add_argument("--group3-report", required=True)
    parser.add_argument("--residual-weight", type=float, default=0.125)
    parser.add_argument("--movement-cap-ratio", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--base-public-score", type=float, default=0.6440998116)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promotion_tier": report["promotion_tier"],
                "local_full_score_deltas": report[
                    "local_full_score_deltas"
                ],
                "expected_macro_score_delta": report[
                    "expected_macro_score_delta_if_2024_transfers"
                ],
                "projected_public_score": report[
                    "projected_public_score_if_local_delta_transfers"
                ],
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
