"""Audit score scale and residual calibration headroom of public 1158 OOF.

Same-axis oracle transformations are diagnostics, never candidates.  The only
transfer estimates fit on a labelled source slice and apply unchanged to a
future audit slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import (
    brier_decomposition,
    brier_score,
    brier_skill_score_unclipped,
    calibration_intercept_slope,
)
from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes


PUBLIC_CHAMPION = 1158.0745556751
PUBLIC_TARGET = 1170.0


def fit_brier_calibrator(target: np.ndarray, parent: np.ndarray, method: str) -> dict[str, float]:
    """Fit a low-dimensional Brier calibrator on source rows only."""

    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(parent, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("calibration arrays must be non-empty, aligned 1-D arrays")
    if method == "additive":
        return {"method": method, "intercept": float(np.mean(y - p)), "slope": 1.0}
    if method == "affine":
        design = np.column_stack([np.ones(len(p), dtype=np.float64), p])
        coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
        return {
            "method": method,
            "intercept": float(coefficient[0]),
            "slope": float(coefficient[1]),
        }
    raise ValueError(f"unknown calibration method: {method}")


def apply_brier_calibrator(parent: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    p = np.asarray(parent, dtype=np.float64)
    return np.clip(float(spec["intercept"]) + float(spec["slope"]) * p, 0.001, 0.999)


def _absolute_diagnostics(frame: pd.DataFrame, parent: np.ndarray) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    intercept, slope = calibration_intercept_slope(target, parent)
    decompositions = {
        str(bins): brier_decomposition(target, parent, n_bins=bins)
        for bins in (10, 20, 50)
    }
    return {
        "rows": int(len(frame)),
        "target_rate": float(target.mean()),
        "prediction_mean": float(np.mean(parent)),
        "mean_gap_prediction_minus_target": float(np.mean(parent) - target.mean()),
        "brier": brier_score(target, parent),
        "unclipped_bss_equivalent": brier_skill_score_unclipped(target, parent),
        "calibration_intercept_logit": intercept,
        "calibration_slope_logit": slope,
        "brier_decomposition": decompositions,
    }


def _evaluate_calibrator(
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    spec: dict[str, float],
) -> dict[str, object]:
    candidate = apply_brier_calibrator(audit_parent, spec)
    active = np.ones(len(audit_frame), dtype=bool)
    result = diagnostics(audit_frame, audit_parent, candidate, active)
    result.update(spec)
    return result


def _oracle_group_offset(
    frame: pd.DataFrame,
    parent: np.ndarray,
    columns: tuple[str, ...],
    *,
    prior: float,
) -> tuple[np.ndarray, dict[str, object]]:
    local = frame.loc[:, list(columns)].copy()
    local["residual"] = frame["target"].to_numpy(np.float64) - np.asarray(parent)
    grouped = local.groupby(list(columns), observed=True)["residual"].agg(["sum", "size"])
    grouped["offset"] = grouped["sum"] / (grouped["size"] + float(prior))
    offsets = local.merge(
        grouped[["offset"]].reset_index(),
        on=list(columns),
        how="left",
        sort=False,
        validate="many_to_one",
    )["offset"].fillna(0.0).to_numpy(np.float64)
    candidate = np.clip(np.asarray(parent, dtype=np.float64) + offsets, 0.001, 0.999)
    return candidate, {
        "groups": int(len(grouped)),
        "prior": float(prior),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
    }


def run(project: Path, current_oof_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    frames = {
        "late_2023": axes["selection_late_2023"],
        "full_2024": axes["outer_full_2024"],
        "late_2024": axes["replication_late_2024"],
    }
    caches = {
        "late_2023": np.load(current_oof_dir / "selection_late_2023.npz"),
        "full_2024": np.load(current_oof_dir / "outer_full_2024.npz"),
        "late_2024": np.load(current_oof_dir / "replication_late_2024.npz"),
    }
    parents: dict[str, np.ndarray] = {}
    absolute: dict[str, object] = {}
    oracle: dict[str, object] = {}
    for name, frame in frames.items():
        cache = caches[name]
        if not np.array_equal(cache["target"], frame["target"].to_numpy(np.float64)):
            raise ValueError(f"target parity failure: {name}")
        parent = cache["final_gate_parent"].astype(np.float64)
        parents[name] = parent
        absolute[name] = _absolute_diagnostics(frame, parent)
        oracle[name] = {}
        target = frame["target"].to_numpy(np.float64)
        for method in ("additive", "affine"):
            spec = fit_brier_calibrator(target, parent, method)
            oracle[name][method] = _evaluate_calibrator(frame, parent, spec)
        group_specs = {
            "domain": (("domain3",), 2000.0),
            "month": (("game_month",), 5000.0),
            "domain_count": (("domain3", "count_state"), 2000.0),
        }
        for label, (columns, prior) in group_specs.items():
            candidate, metadata = _oracle_group_offset(
                frame, parent, columns, prior=prior
            )
            result = diagnostics(frame, parent, candidate, np.ones(len(frame), dtype=bool))
            oracle[name][label] = {**metadata, **result}

    full24 = frames["full_2024"]
    early_mask = full24["game_month"].le(7).to_numpy()
    source_frames = {
        "late23_to_full24": (frames["late_2023"], parents["late_2023"], frames["full_2024"], parents["full_2024"]),
        "early24_to_late24": (
            full24.loc[early_mask].reset_index(drop=True),
            parents["full_2024"][early_mask],
            frames["late_2024"],
            parents["late_2024"],
        ),
    }
    transfer: dict[str, object] = {}
    for name, (source_frame, source_parent, audit_frame, audit_parent) in source_frames.items():
        transfer[name] = {}
        for method in ("additive", "affine"):
            spec = fit_brier_calibrator(
                source_frame["target"].to_numpy(np.float64), source_parent, method
            )
            transfer[name][method] = _evaluate_calibrator(audit_frame, audit_parent, spec)

    gap = PUBLIC_TARGET - PUBLIC_CHAMPION
    score_scale = []
    for rate in (0.45, 0.4861049201797189, 0.50, 0.55):
        reference = rate * (1.0 - rate)
        score_scale.append(
            {
                "assumed_test_rate": rate,
                "reference_brier": reference,
                "brier_reduction_needed_for_1170": gap / 100000.0 * reference,
            }
        )
    result = {
        "protocol": "V75_POST_1158_EVALUATION_HEADROOM_AUDIT_V1",
        "public_champion": PUBLIC_CHAMPION,
        "public_target": PUBLIC_TARGET,
        "public_gap": gap,
        "score_scale": score_scale,
        "absolute_oof": absolute,
        "same_axis_oracle_diagnostic_only": oracle,
        "source_only_forward_calibration": transfer,
        "selection_warning": (
            "2024 has been repeatedly inspected; all 2024 results are development-contaminated "
            "confirmation, not independent holdout evidence"
        ),
        "test_csv_read": False,
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
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
