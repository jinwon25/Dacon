"""Extend the paired workload-H1 audit to untouched 2020/2021 origins.

The exact v203 recipe is reused with seed 42.  Only the two earlier forward
origins are newly fit; 2022--2024 predictions are read from the preregistered
v203 run.  A fixed 25% component dose is the primary long-horizon check.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import affine, metrics
from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.archive.v203_h1_workload_feature_screen import MODEL_CONFIG
from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    strict_axis,
)
from src.champion.v130_hoo_independent_oof_blend import post4
from src.champion.v131_hoo_h1_independent_oof import _fit_year, _prepare_features


PROTOCOL = "V207_H1_WORKLOAD_EARLY_ORIGIN_AUDIT_V1"
YEARS = (2020, 2021, 2022, 2023, 2024)
NEW_YEARS = (2020, 2021)
PRIMARY_SCALE = 0.25
DIAGNOSTIC_SCALES = (0.10, 0.25, 0.50)


def origin_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.50
        and result["worst_month_gain"] > -5.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_fits": True,
        "paired_same_h1_recipe_and_seed": True,
        "fixed_primary_scale": True,
        "early_origins_not_used_by_v203_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    external_root: Path,
    baseline_checkpoint_dir: Path,
    v203_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    workload = workload_feature_frame(train)
    augmented_features = list(h1_features) + list(workload.columns)
    train = train.copy()
    for column in workload.columns:
        train[column] = workload[column].to_numpy(np.float32)
    del workload
    gc.collect()

    augmented_raw: dict[int, np.ndarray] = {}
    for year in YEARS:
        if year in NEW_YEARS:
            checkpoint = output_dir / f"augmented_h1_year{year}_seed42.npy"
            if checkpoint.exists():
                augmented_raw[year] = np.load(
                    checkpoint, allow_pickle=False
                ).astype(np.float64)
            else:
                augmented_raw[year] = _fit_year(
                    train, target, season, year, augmented_features,
                    MODEL_CONFIG, "workload-H1-early-seed42",
                )
                np.save(checkpoint, augmented_raw[year], allow_pickle=False)
        else:
            augmented_raw[year] = np.load(
                v203_dir / f"augmented_h1_year{year}_seed42.npy",
                allow_pickle=False,
            ).astype(np.float64)

    details: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for scale in DIAGNOSTIC_SCALES:
        details[str(scale)] = {}
        for year in YEARS:
            audit = season == year
            frame = train.loc[audit].reset_index(drop=True)
            correction = post4(
                train.loc[season < year].reset_index(drop=True), frame
            )
            baseline_raw = np.load(
                baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
                allow_pickle=False,
            ).astype(np.float64)
            baseline = affine(baseline_raw + correction)
            augmented = affine(augmented_raw[year] + correction)
            if not (len(frame) == len(baseline) == len(augmented)):
                raise ValueError(f"early-origin alignment mismatch: {year}")
            axis = strict_axis(frame, target[audit], baseline)
            candidate = apply_component_delta(baseline, augmented, scale)
            result = metrics(axis, baseline, candidate)
            details[str(scale)][str(year)] = result
            rows.append(
                {
                    "scale": scale,
                    "year": year,
                    "gain": result["gain"],
                    "positive_month_fraction": result[
                        "positive_month_fraction"
                    ],
                    "worst_month_gain": result["worst_month_gain"],
                    "gate_passed": origin_gate(result),
                }
            )

    primary = details[str(PRIMARY_SCALE)]
    early_pass = all(origin_gate(primary[str(year)]) for year in NEW_YEARS)
    all_year_positive = all(primary[str(year)]["gain"] > 0.0 for year in YEARS)
    summary = {
        "protocol": PROTOCOL,
        "status": "early_origin_support" if early_pass and all_year_positive else (
            "early_origin_reject" if not early_pass else "recent_origin_sign_reject"
        ),
        "model_config": MODEL_CONFIG,
        "primary_scale": PRIMARY_SCALE,
        "primary": primary,
        "early_origin_gate_passed": early_pass,
        "all_five_year_gains_positive": all_year_positive,
        "eligible_for_multiseed_research": bool(early_pass and all_year_positive),
        "eligible_for_packaging": False,
        "diagnostic_rows": rows,
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
    parser.add_argument("--v203-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.external_root,
        args.baseline_checkpoint_dir, args.v203_dir, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
