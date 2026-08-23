"""Rebase the frozen legal latent-pitch-type direction above public 1158.

The latent pitch-type model uses only previous-season labelled rows and the
current row's official fields at inference.  Its old 2024 audit reported an
all-month-positive gain at weights 0.05 and 0.10, but it predates the true
public-1158 parent and was outside the v58 version sweep.

This audit preserves the original probability-space direction exactly.  The
primary 0.05 weight is frozen from the original report; 0.10 is a documented
sensitivity value, not a new audit-selected recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes


PRIMARY_WEIGHT = 0.05
FROZEN_WEIGHTS = (0.05, 0.10)


def frozen_pitch_type_shift(saved: np.lib.npyio.NpzFile, weight: float) -> np.ndarray:
    incumbent = saved["incumbent"].astype(np.float64)
    student = saved["student_raw"].astype(np.float64)
    if incumbent.shape != student.shape or incumbent.ndim != 1:
        raise ValueError("latent pitch-type artifact shape mismatch")
    return float(weight) * (student - incumbent)


def _evaluate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    shift: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    candidate = np.clip(np.asarray(parent, dtype=np.float64) + shift, 0.001, 0.999)
    active = np.abs(shift) > 0
    result = diagnostics(frame, parent, candidate, active)
    result.update(
        {
            "shift_mean": float(np.mean(shift)),
            "shift_mean_abs": float(np.mean(np.abs(shift))),
            "shift_sd": float(np.std(shift)),
            "shift_max_abs": float(np.max(np.abs(shift))),
        }
    )
    return result, candidate


def run(
    project: Path,
    latent_dir: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    latent_dir = latent_dir.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    current23 = np.load(current_oof_dir / "selection_late_2023.npz")
    current24 = np.load(current_oof_dir / "outer_full_2024.npz")
    current_rep = np.load(current_oof_dir / "replication_late_2024.npz")
    latent23 = np.load(latent_dir / "latent_pitch_type_o2023.npz")
    latent24 = np.load(latent_dir / "latent_pitch_type_o2024.npz")
    late23_mask = latent23["game_month"].astype(np.int16) >= 8
    late24_mask = latent24["game_month"].astype(np.int16) >= 8
    if not np.array_equal(
        latent23["target"].astype(np.float64)[late23_mask],
        late23["target"].to_numpy(np.float64),
    ):
        raise ValueError("latent late-2023 target parity failure")
    if not np.array_equal(
        latent24["target"].astype(np.float64),
        full24["target"].to_numpy(np.float64),
    ):
        raise ValueError("latent full-2024 target parity failure")
    if not np.array_equal(
        latent24["target"].astype(np.float64)[late24_mask],
        replication24["target"].to_numpy(np.float64),
    ):
        raise ValueError("latent late-2024 target parity failure")

    metric_rows: list[dict[str, object]] = []
    primary_predictions: dict[str, np.ndarray] = {}
    for weight in FROZEN_WEIGHTS:
        shift23 = frozen_pitch_type_shift(latent23, weight)[late23_mask]
        shift24 = frozen_pitch_type_shift(latent24, weight)
        axis_specs = {
            "selection_late_2023": (
                late23,
                current23["final_gate_parent"].astype(np.float64),
                shift23,
            ),
            "outer_full_2024": (
                full24,
                current24["final_gate_parent"].astype(np.float64),
                shift24,
            ),
            "replication_late_2024": (
                replication24,
                current_rep["final_gate_parent"].astype(np.float64),
                shift24[late24_mask],
            ),
        }
        for axis_name, (frame, parent, shift) in axis_specs.items():
            result, candidate = _evaluate(frame, parent, shift)
            metric_rows.append(
                {"weight": weight, "axis": axis_name, **result}
            )
            if weight == PRIMARY_WEIGHT:
                primary_predictions[axis_name] = candidate

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "primary_predictions.npz", **primary_predictions)
    primary = metrics.loc[metrics["weight"].eq(PRIMARY_WEIGHT)]
    gates = {
        "all_axis_gains_positive": bool((primary["gain"] > 0).all()),
        "all_month_fractions_at_least_075": bool(
            (primary["positive_month_fraction"] >= 0.75).all()
        ),
        "all_minimum_domains_nonnegative": bool(
            (primary["minimum_domain_gain"] >= 0).all()
        ),
        "all_worst_months_above_minus_05": bool(
            (primary["worst_month_gain"] > -0.5).all()
        ),
    }
    primary_by_axis = {
        row["axis"]: {
            key: row[key]
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "shift_mean_abs",
            )
        }
        for _, row in primary.iterrows()
    }
    result = {
        "protocol": "V71_FROZEN_LATENT_PITCH_TYPE_SHIFT_ABOVE_1158_V1",
        "primary_weight": PRIMARY_WEIGHT,
        "direction": "student_raw_minus_original_v17_incumbent",
        "primary_audits": primary_by_axis,
        "sensitivity": metrics[
            [
                "weight",
                "axis",
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
            ]
        ].to_dict(orient="records"),
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "decision": "promote" if all(gates.values()) else "reject",
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.latent_dir, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
