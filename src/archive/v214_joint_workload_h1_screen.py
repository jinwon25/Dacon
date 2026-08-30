"""Screen joint 1/3/5-game workload disambiguation in exact H1.

Recent success/middle rate pairs identify a minimum common denominator.  In
about one sixth of pitcher appearances that denominator is a divisor of the
true pitch count.  This row-local feature family uses the median implied
pitches per appearance across the 1/3/5-game windows, then lifts each minimum
denominator to the nearest compatible multiple.  Both the original minimum
features and these joint candidates are retained.  Seed 42 is a screen only;
it cannot be packaged without independent seeds.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import affine, metrics
from src.archive.v199_recent_workload_reconstruction import (
    HORIZONS,
    MAX_DENOMINATOR,
    infer_denominators,
    rate_columns,
)
from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.archive.v203_h1_workload_feature_screen import MODEL_CONFIG
from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    strict_axis,
)
from src.champion.v130_hoo_independent_oof_blend import post4
from src.champion.v131_hoo_h1_independent_oof import _fit_year, _prepare_features


PROTOCOL = "V214_JOINT_WORKLOAD_H1_SCREEN_V1"
YEARS = (2022, 2023, 2024)
SCREEN_SCALE = 0.10


def joint_workload_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create label-free joint denominator candidates from a single row."""

    minimum: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}
    success_rate: dict[int, np.ndarray] = {}
    middle_rate: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        success_column, middle_column = rate_columns(horizon)
        success = pd.to_numeric(
            frame[success_column], errors="coerce"
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            frame[middle_column], errors="coerce"
        ).to_numpy(np.float64)
        inferred = infer_denominators(
            pd.Series(success), pd.Series(middle),
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        n = pd.to_numeric(
            inferred["minimum_denominator"], errors="coerce"
        ).fillna(0).to_numpy(np.float64)
        success_rate[horizon] = success
        middle_rate[horizon] = middle
        minimum[horizon] = n
        valid[horizon] = (
            inferred["rounded_rational_fit"].fillna(False).to_numpy(bool)
            & np.isfinite(success)
            & np.isfinite(middle)
            & (n > 0.0)
        )

    implied = np.column_stack([
        np.where(valid[horizon], minimum[horizon] / horizon, np.nan)
        for horizon in HORIZONS
    ])
    valid_count = np.sum(np.isfinite(implied), axis=1)
    safe_implied = implied.copy()
    safe_implied[valid_count == 0, 0] = 0.0
    with np.errstate(all="ignore"):
        reference = np.nanmedian(safe_implied, axis=1)
    reference = np.nan_to_num(reference, nan=0.0)

    output: dict[str, np.ndarray] = {}
    joint: dict[int, np.ndarray] = {}
    career_success = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        frame["asof_pitcher_middle_rate"], errors="coerce"
    ).fillna(0.2).to_numpy(np.float64)
    priors = {1: 20.0, 3: 60.0, 5: 100.0}
    for horizon in HORIZONS:
        n = minimum[horizon]
        multiplier = np.maximum(
            1.0,
            np.rint(
                np.divide(
                    reference * horizon,
                    n,
                    out=np.ones(len(frame), dtype=np.float64),
                    where=n > 0.0,
                )
            ),
        )
        candidate = np.minimum(n * multiplier, MAX_DENOMINATOR[horizon])
        candidate = np.where(valid[horizon], candidate, 0.0)
        joint[horizon] = candidate
        reliability = candidate / (candidate + priors[horizon])
        output[f"joint_prev{horizon}_log_n"] = np.log1p(candidate)
        output[f"joint_prev{horizon}_multiplier"] = np.where(
            valid[horizon], multiplier, 0.0
        )
        output[f"joint_prev{horizon}_reliability"] = reliability
        output[f"joint_prev{horizon}_success_reliable_delta"] = np.where(
            valid[horizon],
            reliability * (success_rate[horizon] - career_success),
            0.0,
        )
        output[f"joint_prev{horizon}_middle_reliable_delta"] = np.where(
            valid[horizon],
            reliability * (middle_rate[horizon] - career_middle),
            0.0,
        )

    n1, n3, n5 = (joint[horizon] for horizon in HORIZONS)
    output["joint_prev3_minus_prev1_log"] = np.log1p(
        np.maximum(n3 - n1, 0.0)
    )
    output["joint_prev5_minus_prev3_log"] = np.log1p(
        np.maximum(n5 - n3, 0.0)
    )
    output["joint_reference_pitches_per_game"] = reference
    implied_range = np.nanmax(
        np.where(np.isfinite(implied), implied, -np.inf), axis=1
    ) - np.nanmin(
        np.where(np.isfinite(implied), implied, np.inf), axis=1
    )
    output["joint_valid_horizon_count"] = valid_count.astype(np.float64)
    output["joint_minimum_per_game_range"] = np.where(
        valid_count > 1, implied_range, 0.0
    )
    return pd.DataFrame(output, index=frame.index, dtype=np.float32)


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_fits": True,
        "row_local_joint_denominator_inference": True,
        "fixed_median_nearest_multiple_rule": True,
        "fixed_seed42_screen": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    external_root: Path,
    baseline_checkpoint_dir: Path,
    existing_workload_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    workload = workload_feature_frame(train)
    joint = joint_workload_features(train)
    feature_names = list(h1_features) + list(workload.columns) + list(joint.columns)
    train = train.copy()
    for feature_frame in (workload, joint):
        for column in feature_frame.columns:
            train[column] = feature_frame[column].to_numpy(np.float32)
    del workload, joint
    gc.collect()

    details: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, float]] = {}
    fitted: list[dict[str, Any]] = []
    for year in YEARS:
        checkpoint = output_dir / f"joint_h1_year{year}_seed42.npy"
        if checkpoint.exists():
            joint_raw = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            reused = True
        else:
            joint_raw = _fit_year(
                train, target, season, year, feature_names, MODEL_CONFIG,
                "joint-workload-H1-seed42",
            )
            np.save(checkpoint, joint_raw, allow_pickle=False)
            reused = False
        baseline_raw = np.load(
            baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        existing_raw = np.load(
            existing_workload_dir / f"augmented_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        audit = season == year
        frame = train.loc[audit].reset_index(drop=True)
        correction = post4(train.loc[season < year].reset_index(drop=True), frame)
        baseline = affine(baseline_raw + correction)
        existing = affine(existing_raw + correction)
        joint_prediction = affine(joint_raw + correction)
        axis = strict_axis(frame, target[audit], baseline)
        old_candidate = apply_component_delta(
            baseline, existing, SCREEN_SCALE
        )
        new_candidate = apply_component_delta(
            baseline, joint_prediction, SCREEN_SCALE
        )
        old_result = metrics(axis, baseline, old_candidate)
        new_result = metrics(axis, baseline, new_candidate)
        details[str(year)] = new_result
        comparisons[str(year)] = {
            "existing_gain": float(old_result["gain"]),
            "joint_gain": float(new_result["gain"]),
            "incremental_gain": float(
                new_result["gain"] - old_result["gain"]
            ),
            "raw_prediction_correlation": float(
                np.corrcoef(existing_raw, joint_raw)[0, 1]
            ),
        }
        fitted.append({
            "year": year,
            "rows": len(joint_raw),
            "reused": reused,
            "checkpoint": str(checkpoint),
        })

    all_positive = all(details[str(year)]["gain"] > 0.0 for year in YEARS)
    source_month_pass = all(
        details[str(year)]["positive_month_fraction"] >= 0.70
        and details[str(year)]["worst_month_gain"] > -5.0
        for year in (2022, 2023)
    )
    incremental_years = sum(
        comparisons[str(year)]["incremental_gain"] > 0.0 for year in YEARS
    )
    eligible = bool(all_positive and source_month_pass and incremental_years >= 2)
    summary = {
        "protocol": PROTOCOL,
        "status": "confirm_multiseed" if eligible else "screen_reject",
        "screen_scale": SCREEN_SCALE,
        "model_config": MODEL_CONFIG,
        "strict_years": details,
        "comparison_to_v209_workload": comparisons,
        "all_year_gains_positive": all_positive,
        "source_month_gate_passed": source_month_pass,
        "incremental_positive_years": int(incremental_years),
        "eligible_for_multiseed_confirmation": eligible,
        "eligible_for_packaging": False,
        "fitted": fitted,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--existing-workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.external_root,
        args.baseline_checkpoint_dir, args.existing_workload_dir,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
