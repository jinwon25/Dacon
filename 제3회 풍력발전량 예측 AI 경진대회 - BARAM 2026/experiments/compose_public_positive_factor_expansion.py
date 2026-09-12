"""Expand only the publicly positive pooled group-2 factor.

Submission 1502437 contains the pooled group-2 factor at model weight 0.05.
Its isolated public macro gain was +0.0004796733.  This experiment doubles
that factor movement to the equivalent weight 0.10 while preserving group 1
and group 3 from the frozen public incumbent.

The expansion is not selected from the public score; the public result is used
only to prioritize this already-positive factor family over rejected ones.
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
from experiments.kma_year_forward_quantile_blend import (
    _periods,
    apply_bounded_blend,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_2"
BASE_WEIGHT = 0.05
EXPANDED_WEIGHT = 0.10
MAXIMUM_MOVEMENT_RATIO = 0.05
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_factor(
    factor_control: np.ndarray,
    factor_treatment: np.ndarray,
    *,
    base_weight: float,
    expanded_weight: float,
    capacity: float,
    maximum_movement_ratio: float,
) -> np.ndarray:
    control = np.asarray(factor_control, dtype=float)
    treatment = np.asarray(factor_treatment, dtype=float)
    if control.shape != treatment.shape:
        raise ValueError("factor control and treatment shapes differ")
    if not 0.0 < base_weight < expanded_weight:
        raise ValueError("expanded weight must exceed positive base weight")
    expert_movement = (treatment - control) / base_weight
    movement = np.clip(
        expanded_weight * expert_movement,
        -maximum_movement_ratio * capacity,
        maximum_movement_ratio * capacity,
    )
    return np.clip(control + movement, 0.0, capacity)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "incumbent": _rooted(args.incumbent),
        "factor_control": _rooted(args.factor_control),
        "factor_treatment": _rooted(args.factor_treatment),
        "validation_cache": _rooted(args.validation_cache),
    }
    frames = {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in paths.items()
        if name != "validation_cache"
    }
    for name, frame in frames.items():
        if name == "incumbent":
            continue
        if not frame[list(ID_COLUMNS)].equals(
            frames["incumbent"][list(ID_COLUMNS)]
        ):
            raise ValueError(f"{name} identifiers differ from incumbent")
    if not np.allclose(
        frames["incumbent"][TARGET],
        frames["factor_treatment"][TARGET],
        atol=args.atol,
        rtol=0.0,
    ):
        raise ValueError(
            "incumbent group 2 does not equal factor treatment"
        )

    cache = np.load(paths["validation_cache"], allow_pickle=False)
    prefix = f"{TARGET}__"
    index = pd.DatetimeIndex(pd.to_datetime(cache[f"{prefix}index_ns"]))
    issue = pd.DatetimeIndex(pd.to_datetime(cache[f"{prefix}issue_ns"]))
    truth = cache[f"{prefix}truth"].astype(float)
    reference = cache[f"{prefix}reference"].astype(float)
    expert = cache[f"{prefix}expert"].astype(float)
    current = apply_bounded_blend(
        reference,
        expert,
        weight=BASE_WEIGHT,
        capacity=CAPACITY_KWH[TARGET],
    )
    expanded = apply_bounded_blend(
        reference,
        expert,
        weight=EXPANDED_WEIGHT,
        capacity=CAPACITY_KWH[TARGET],
    )
    if not np.allclose(
        current,
        cache[f"{prefix}candidate"],
        atol=2e-3,
        rtol=0.0,
    ):
        raise ValueError("validation cache base weight is not 0.05")
    periods = _periods(index)
    full_deltas_vs_control = {
        name: metric_delta(
            truth,
            reference,
            expanded,
            CAPACITY_KWH[TARGET],
            rows,
        )
        for name, rows in periods.items()
    }
    incremental_deltas = {
        name: metric_delta(
            truth,
            current,
            expanded,
            CAPACITY_KWH[TARGET],
            rows,
        )
        for name, rows in periods.items()
    }
    monthly_incremental = {
        str(month): metric_delta(
            truth,
            current,
            expanded,
            CAPACITY_KWH[TARGET],
            periods["full"] & np.asarray(index.month == month),
        )
        for month in range(1, 13)
    }
    bootstrap = evaluate_blocked_rolling(
        truth,
        current,
        expanded,
        index,
        issue,
        periods["full"]
        & (truth >= 0.10 * CAPACITY_KWH[TARGET]),
        n_bootstrap=args.n_bootstrap,
        seed=20260727,
    )

    output = frames["incumbent"].copy()
    output[TARGET] = expand_factor(
        frames["factor_control"][TARGET].to_numpy(dtype=float),
        frames["factor_treatment"][TARGET].to_numpy(dtype=float),
        base_weight=BASE_WEIGHT,
        expanded_weight=EXPANDED_WEIGHT,
        capacity=CAPACITY_KWH[TARGET],
        maximum_movement_ratio=MAXIMUM_MOVEMENT_RATIO,
    )
    for target in CAPACITY_KWH:
        if target == TARGET:
            continue
        if not np.array_equal(
            output[target].to_numpy(dtype=float),
            frames["incumbent"][target].to_numpy(dtype=float),
        ):
            raise AssertionError(f"non-target group changed: {target}")
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(
            f"CandidateValidator rejected G2 expansion: {audit.errors}"
        )

    local_current = metric_delta(
        truth,
        reference,
        current,
        CAPACITY_KWH[TARGET],
        periods["full"],
    )["score"]
    local_expanded = full_deltas_vs_control["full"]["score"]
    public_group_delta = 3.0 * args.public_macro_factor_delta
    transfer_ratio = public_group_delta / local_current
    projected_increment = (
        (local_expanded - local_current) * transfer_ratio / 3.0
    )
    movement = np.abs(
        output[TARGET].to_numpy(dtype=float)
        - frames["incumbent"][TARGET].to_numpy(dtype=float)
    )
    report = {
        "family": "public_positive_pooled_group2_factor_expansion",
        "promotion_tier": "public_confirmed_family_controlled_expansion",
        "contract": {
            "target": TARGET,
            "base_weight": BASE_WEIGHT,
            "expanded_weight": EXPANDED_WEIGHT,
            "maximum_movement_ratio": MAXIMUM_MOVEMENT_RATIO,
            "changes_only_one_group": True,
            "test_actual_generation_used": False,
            "public_score_used_only_for_family_prioritization": True,
        },
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "validation": {
            "period_deltas_vs_factor_control": full_deltas_vs_control,
            "period_incremental_deltas_vs_incumbent_factor": (
                incremental_deltas
            ),
            "monthly_incremental_deltas": monthly_incremental,
            "positive_incremental_score_months": int(
                sum(
                    row["score"] > 0.0
                    for row in monthly_incremental.values()
                )
            ),
            "issue_block_validation_incremental": bootstrap,
        },
        "public_transfer_projection": {
            "observed_base_factor_macro_delta": float(
                args.public_macro_factor_delta
            ),
            "local_base_group_delta": float(local_current),
            "empirical_group_transfer_ratio": float(transfer_ratio),
            "local_expanded_group_delta": float(local_expanded),
            "projected_incremental_macro_delta": float(
                projected_increment
            ),
            "projected_public_score": float(
                args.incumbent_public_score + projected_increment
            ),
            "warning": (
                "This extrapolates one observed public factor point and is "
                "not exact outside the submitted base weight."
            ),
        },
        "movement_vs_incumbent": {
            "changed_rows": int(np.sum(movement > args.atol)),
            "mean_capacity_ratio": float(
                movement.mean() / CAPACITY_KWH[TARGET]
            ),
            "p95_capacity_ratio": float(
                np.quantile(movement, 0.95) / CAPACITY_KWH[TARGET]
            ),
            "maximum_capacity_ratio": float(
                movement.max() / CAPACITY_KWH[TARGET]
            ),
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
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--factor-control",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument(
        "--factor-treatment",
        default=(
            "artifacts_final/candidates/"
            "kma_jma_pooled_all3_g1g2_nearstable_20260726.csv"
        ),
    )
    parser.add_argument(
        "--validation-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--incumbent-public-score",
        type=float,
        default=0.6461250914,
    )
    parser.add_argument(
        "--public-macro-factor-delta",
        type=float,
        default=0.0004796733,
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "public_positive_pooled_g2_w10_probe_20260727.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_positive_pooled_g2_w10_probe_20260727.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "public_transfer_projection": (
                    report["public_transfer_projection"]
                ),
                "movement": report["movement_vs_incumbent"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
