"""Prune a stable multi-resolution KMA blend with physical regime gates.

Thresholds are estimated and selected on 2024 Q1 only.  Q2, H2, calendar
months, and complete issue blocks remain confirmation evidence.  The router
can only remove part of an already bounded candidate movement; it cannot
create a new extrapolative correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.kma_year_forward_quantile_blend import (
    H2_START,
    Q2_START,
    VALIDATION_END,
    load_context_features,
    metric_delta,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.25, 0.50, 0.75)
MINIMUM_COVERAGE = 0.20
MAXIMUM_COVERAGE = 0.95


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_regime_features(
    regional: pd.DataFrame,
    local: pd.DataFrame,
    reference: np.ndarray,
    current: np.ndarray,
    capacity: float,
) -> pd.DataFrame:
    """Construct deterministic multi-model disagreement and movement features."""
    if not regional.index.equals(local.index):
        raise ValueError("regional and local context indexes differ")
    if len(regional) != len(reference) or len(reference) != len(current):
        raise ValueError("regime features and validation predictions do not align")
    output: dict[str, np.ndarray] = {}
    for level in ("10", "850", "700"):
        regional_u = regional[f"kma_um_ctx_u{level}_r0"].to_numpy(dtype=float)
        regional_v = regional[f"kma_um_ctx_v{level}_r0"].to_numpy(dtype=float)
        local_u = local[f"kma_um_ctx_u{level}_r0"].to_numpy(dtype=float)
        local_v = local[f"kma_um_ctx_v{level}_r0"].to_numpy(dtype=float)
        regional_speed = np.hypot(regional_u, regional_v)
        local_speed = np.hypot(local_u, local_v)
        denominator = np.maximum(regional_speed * local_speed, 1e-6)
        output[f"speed_delta_{level}"] = local_speed - regional_speed
        output[f"abs_speed_delta_{level}"] = np.abs(
            output[f"speed_delta_{level}"]
        )
        output[f"vector_disagreement_{level}"] = np.hypot(
            local_u - regional_u,
            local_v - regional_v,
        )
        output[f"direction_cosine_{level}"] = np.clip(
            (regional_u * local_u + regional_v * local_v) / denominator,
            -1.0,
            1.0,
        )
    for level in ("850", "700"):
        output[f"local_shear10_{level}"] = local[
            f"kma_um_ctx_shear10_{level}_r0"
        ].to_numpy(dtype=float)
        output[f"regional_shear10_{level}"] = regional[
            f"kma_um_ctx_shear10_{level}_r0"
        ].to_numpy(dtype=float)
    movement = (np.asarray(current) - np.asarray(reference)) / float(capacity)
    output["signed_candidate_movement"] = movement
    output["absolute_candidate_movement"] = np.abs(movement)
    output["reference_ratio"] = np.asarray(reference, dtype=float) / float(capacity)
    frame = pd.DataFrame(output, index=regional.index, dtype="float64")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("regime features are incomplete")
    return frame


def make_gate(
    values: np.ndarray,
    development: np.ndarray,
    *,
    quantile: float,
    direction: str,
) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(values[development], quantile))
    if direction == "high":
        return values >= threshold, threshold
    if direction == "low":
        return values <= threshold, threshold
    raise ValueError(f"unsupported gate direction: {direction}")


def route_candidate(
    reference: np.ndarray,
    current: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """Retain the current bounded correction only inside the physical gate."""
    output = np.asarray(reference, dtype=float).copy()
    output[np.asarray(gate, dtype=bool)] = np.asarray(current, dtype=float)[gate]
    return output


def _periods(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "q1": np.asarray(index < Q2_START),
        "q2": np.asarray((index >= Q2_START) & (index < H2_START)),
        "h2": np.asarray(
            (index >= H2_START) & (index < VALIDATION_END)
        ),
        "full": np.asarray(index < VALIDATION_END),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    target = args.target
    capacity = CAPACITY_KWH[target]
    cache_path = _rooted(args.validation_cache)
    cache = np.load(cache_path, allow_pickle=False)
    prefix = f"{target}__"
    index = pd.DatetimeIndex(pd.to_datetime(cache[f"{prefix}index_ns"]))
    issue = pd.DatetimeIndex(pd.to_datetime(cache[f"{prefix}issue_ns"]))
    truth = cache[f"{prefix}truth"].astype(float)
    reference = cache[f"{prefix}reference"].astype(float)
    current = cache[f"{prefix}candidate"].astype(float)
    regional, _ = load_context_features(_rooted(args.regional_context))
    local, _ = load_context_features(_rooted(args.local_context))
    regional = regional.reindex(index)
    local = local.reindex(index)
    for name, frame in (("regional", regional), ("local", local)):
        missing_rows = frame.isna().any(axis=1)
        if int(missing_rows.sum()) > 1:
            raise ValueError(f"{name} context has more than one alignment gap")
    # The exact-driver OOF contains the 2025-01-01 00:00 boundary row while
    # annual context files stop just before it.  It is outside every validation
    # period; a one-row carry-forward keeps feature construction total.
    regional = regional.ffill().bfill()
    local = local.ffill().bfill()
    features = build_regime_features(
        regional,
        local,
        reference,
        current,
        capacity,
    )
    periods = _periods(index)

    records: list[dict[str, Any]] = []
    candidates: dict[tuple[str, float, str], np.ndarray] = {}
    gates: dict[tuple[str, float, str], np.ndarray] = {}
    for feature in features:
        values = features[feature].to_numpy(dtype=float)
        for quantile in QUANTILES:
            for direction in ("low", "high"):
                gate, threshold = make_gate(
                    values,
                    periods["q1"],
                    quantile=quantile,
                    direction=direction,
                )
                coverage = float(np.mean(gate[periods["q1"]]))
                if not MINIMUM_COVERAGE <= coverage <= MAXIMUM_COVERAGE:
                    continue
                candidate = route_candidate(reference, current, gate)
                incremental = metric_delta(
                    truth,
                    current,
                    candidate,
                    capacity,
                    periods["q1"],
                )
                total = metric_delta(
                    truth,
                    reference,
                    candidate,
                    capacity,
                    periods["q1"],
                )
                key = (feature, quantile, direction)
                candidates[key] = candidate
                gates[key] = gate
                records.append(
                    {
                        "feature": feature,
                        "quantile": quantile,
                        "direction": direction,
                        "threshold": threshold,
                        "q1_coverage": coverage,
                        "q1_incremental": incremental,
                        "q1_total": total,
                        "eligible": bool(
                            min(incremental.values()) > 0.0
                            and min(total.values()) > 0.0
                        ),
                    }
                )
    eligible = [record for record in records if record["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["q1_incremental"]["score"],
                min(row["q1_incremental"].values()),
                row["q1_coverage"],
            ),
        )
        if eligible
        else None
    )

    result: dict[str, Any] = {
        "family": "kma_multiresolution_physical_regime_router",
        "target": target,
        "sources": {
            "validation_cache": cache_path.relative_to(ROOT).as_posix(),
            "regional_context": _rooted(args.regional_context)
            .relative_to(ROOT)
            .as_posix(),
            "local_context": _rooted(args.local_context)
            .relative_to(ROOT)
            .as_posix(),
        },
        "contract": {
            "selection": "2024 Q1 only",
            "confirmation": "2024 Q2/H2, months, complete issue blocks",
            "action": "remove, never amplify, the existing bounded movement",
            "quantiles": list(QUANTILES),
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
        },
        "records": records,
        "selected": selected,
        "promotion": {"stable": False, "reason": "no eligible Q1 router"},
        "projected_public_score": float(args.base_public_projection),
    }
    if selected is not None:
        key = (
            selected["feature"],
            float(selected["quantile"]),
            selected["direction"],
        )
        candidate = candidates[key]
        gate = gates[key]
        incremental = {
            name: metric_delta(
                truth,
                current,
                candidate,
                capacity,
                rows,
            )
            for name, rows in periods.items()
        }
        total = {
            name: metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                rows,
            )
            for name, rows in periods.items()
        }
        monthly = {
            str(month): metric_delta(
                truth,
                current,
                candidate,
                capacity,
                periods["full"] & np.asarray(index.month == month),
            )
            for month in range(1, 13)
        }
        bootstraps = {
            name: evaluate_blocked_rolling(
                truth,
                current,
                candidate,
                index,
                issue,
                rows & (truth >= 0.10 * capacity),
                n_bootstrap=args.n_bootstrap,
                seed=20260726,
            )
            for name, rows in {
                "h2": periods["h2"],
                "full": periods["full"],
            }.items()
        }
        positive_months = int(
            sum(value["score"] > 0.0 for value in monthly.values())
        )
        confirmation_gates = {
            "q2_incremental_components_positive": (
                min(incremental["q2"].values()) > 0.0
            ),
            "h2_incremental_components_positive": (
                min(incremental["h2"].values()) > 0.0
            ),
            "full_incremental_components_positive": (
                min(incremental["full"].values()) > 0.0
            ),
            "full_total_components_positive": (
                min(total["full"].values()) > 0.0
            ),
            "positive_incremental_month_fraction_at_least_75pct": (
                positive_months / 12.0 >= 0.75
            ),
            "h2_bootstrap_q05_positive": (
                bootstraps["h2"]["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "h2_bootstrap_positive_fraction_at_least_90pct": (
                bootstraps["h2"]["issue_block_bootstrap"][
                    "positive_fraction"
                ]
                >= 0.90
            ),
            "full_bootstrap_q05_positive": (
                bootstraps["full"]["issue_block_bootstrap"]["q05"] > 0.0
            ),
        }
        stable = bool(all(confirmation_gates.values()))
        result.update(
            {
                "period_incremental": incremental,
                "period_total": total,
                "monthly_incremental": monthly,
                "positive_incremental_months": positive_months,
                "issue_block_validation": bootstraps,
                "movement": {
                    "retained_rows": int(gate.sum()),
                    "retained_fraction": float(gate.mean()),
                    "changed_vs_current_rows": int(
                        np.sum(np.abs(candidate - current) > 1e-6)
                    ),
                },
                "confirmation_gates": confirmation_gates,
                "promotion": {
                    "stable": stable,
                    "reason": (
                        "all preregistered confirmation gates passed"
                        if stable
                        else "at least one confirmation gate failed"
                    ),
                },
                "projected_public_score": float(
                    args.base_public_projection
                    + incremental["full"]["score"] / 3.0
                ),
            }
        )

    output_path = _rooted(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=tuple(CAPACITY_KWH),
        required=True,
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument(
        "--regional-context",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/features.csv"
        ),
    )
    parser.add_argument("--local-context", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--base-public-projection", type=float, default=0.647015282)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "target": report["target"],
                "selected": report["selected"],
                "promotion": report["promotion"],
                "projected_public_score": report["projected_public_score"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
