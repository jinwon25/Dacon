from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"


def extrapolate_submission(
    anchor: np.ndarray,
    direction: np.ndarray,
    factor: float,
    capacity: float,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if anchor.shape != direction.shape:
        raise ValueError("anchor and direction must have the same shape")
    return np.clip(anchor + factor * (direction - anchor), 0.0, capacity)


def recover_member(
    blended: np.ndarray,
    base: np.ndarray,
    member_weight: float,
) -> np.ndarray:
    if not 0.0 < member_weight <= 1.0:
        raise ValueError("member_weight must be in (0, 1]")
    blended = np.asarray(blended, dtype=float)
    base = np.asarray(base, dtype=float)
    if blended.shape != base.shape:
        raise ValueError("blended and base must have the same shape")
    return (blended - (1.0 - member_weight) * base) / member_weight


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(source) & np.isfinite(target)
    if mask.sum() < 2 or np.std(source[mask]) == 0.0:
        raise ValueError("Affine alignment requires at least two varying finite rows")
    slope, intercept = np.polyfit(source[mask], target[mask], 1)
    return float(slope), float(intercept)


def _read_target(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if TARGET not in frame:
        raise ValueError(f"{path} does not contain {TARGET}")
    return frame[TARGET].to_numpy(dtype=float)


def audit_and_align(
    blend_path: Path,
    calibration_path: Path,
    pre_cross_path: Path,
    rebuilt_cache_path: Path,
    output_dir: Path,
    extrapolation: float = 1.15,
    scada_weight: float = 0.05,
) -> dict[str, object]:
    capacity = CAPACITY_KWH[TARGET]
    blend = _read_target(blend_path)
    calibration = _read_target(calibration_path)
    pre_cross = _read_target(pre_cross_path)
    over = extrapolate_submission(calibration, blend, extrapolation, capacity)
    historical_scada = recover_member(pre_cross, over, scada_weight)

    cache = np.load(rebuilt_cache_path, allow_pickle=True)
    rebuilt_test = cache[f"{TARGET}__test_matrix"].astype(float)
    if rebuilt_test.ndim != 2 or rebuilt_test.shape[1] != 1:
        raise ValueError("Expected a single rebuilt SCADA candidate")
    rebuilt_test = rebuilt_test[:, 0]
    if len(rebuilt_test) != len(historical_scada):
        raise ValueError("Rebuilt and historical test rows do not match")
    slope, intercept = fit_affine(rebuilt_test, historical_scada)
    aligned_test = np.clip(slope * rebuilt_test + intercept, 0.0, capacity)

    rebuilt_valid = cache[f"{TARGET}__valid_matrix"].astype(float)[:, 0]
    aligned_valid = np.clip(slope * rebuilt_valid + intercept, 0.0, capacity)
    truth = cache[f"{TARGET}__valid_truth"].astype(float)
    metric = evaluate_group(truth, aligned_valid, capacity)

    raw_difference = rebuilt_test - historical_scada
    aligned_difference = aligned_test - historical_scada
    report: dict[str, object] = {
        "target": TARGET,
        "lineage": {
            "over": f"calibration + {extrapolation} * (blend - calibration), clipped",
            "scada_blend_weight": scada_weight,
            "pre_cross_source": str(pre_cross_path),
        },
        "alignment": {"slope": slope, "intercept": intercept},
        "test_comparison": {
            "correlation_raw": float(np.corrcoef(rebuilt_test, historical_scada)[0, 1]),
            "raw_mae_kwh": float(np.mean(np.abs(raw_difference))),
            "aligned_mae_kwh": float(np.mean(np.abs(aligned_difference))),
            "aligned_p95_absolute_difference_kwh": float(
                np.quantile(np.abs(aligned_difference), 0.95)
            ),
            "aligned_max_absolute_difference_kwh": float(
                np.max(np.abs(aligned_difference))
            ),
        },
        "aligned_validation_metric": metric.to_dict(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "scada_group3_aligned_cache.npz",
        valid_index_ns=cache[f"{TARGET}__valid_index_ns"],
        valid_truth=truth.astype("float32"),
        valid_prediction=aligned_valid.astype("float32"),
        test_index_ns=cache["test_index_ns"],
        test_prediction=aligned_test.astype("float32"),
        historical_test_prediction=historical_scada.astype("float32"),
        affine=np.asarray([slope, intercept], dtype="float64"),
    )
    (output_dir / "lineage_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", default="submissions/archive/blend_v1.csv")
    parser.add_argument(
        "--calibration",
        default="submissions/archive/hybrid_lgbm_cat_g3_cal125.csv",
    )
    parser.add_argument("--pre-cross", default="artifacts_cross_group/base_pre_cross.csv")
    parser.add_argument(
        "--rebuilt-cache",
        default="artifacts_scada_stack_hist/prediction_cache.npz",
    )
    parser.add_argument("--output-dir", default="artifacts_oof_lineage")
    parser.add_argument("--extrapolation", type=float, default=1.15)
    parser.add_argument("--scada-weight", type=float, default=0.05)
    args = parser.parse_args()
    report = audit_and_align(
        Path(args.blend),
        Path(args.calibration),
        Path(args.pre_cross),
        Path(args.rebuilt_cache),
        Path(args.output_dir),
        extrapolation=args.extrapolation,
        scada_weight=args.scada_weight,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
