"""Fit seed-43/44 twins for the workload-augmented exact H1 recipe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.archive.v203_h1_workload_feature_screen import MODEL_CONFIG
from src.champion.v131_catboost_h1_independent_oof import _fit_year, _prepare_features


PROTOCOL = "V208_H1_WORKLOAD_MULTISEED_FIT_V1"
SEEDS = (43, 44)
YEARS = (2022, 2023, 2024)


def model_config(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    return {**MODEL_CONFIG, "seed": seed}


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_fits": True,
        "same_recipe_as_seed42": True,
        "fixed_confirmation_seeds": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    workload = workload_feature_frame(train)
    augmented_features = list(h1_features) + list(workload.columns)
    train = train.copy()
    for column in workload.columns:
        train[column] = workload[column].to_numpy(np.float32)
    del workload
    gc.collect()

    fitted: list[dict[str, Any]] = []
    for seed in SEEDS:
        config = model_config(seed)
        for year in YEARS:
            checkpoint = output_dir / f"augmented_h1_year{year}_seed{seed}.npy"
            if checkpoint.exists():
                prediction = np.load(checkpoint, allow_pickle=False)
                expected = int(np.sum(season == year))
                if len(prediction) != expected:
                    raise ValueError(
                        f"multiseed checkpoint length mismatch: {year}/{seed}"
                    )
                reused = True
            else:
                prediction = _fit_year(
                    train, target, season, year, augmented_features, config,
                    f"workload-H1-seed{seed}",
                )
                np.save(checkpoint, prediction, allow_pickle=False)
                reused = False
            fitted.append(
                {
                    "year": year,
                    "seed": seed,
                    "rows": len(prediction),
                    "reused": reused,
                    "checkpoint": str(checkpoint),
                }
            )
    summary = {
        "protocol": PROTOCOL,
        "status": "fit_complete",
        "seeds": list(SEEDS),
        "years": list(YEARS),
        "base_model_config": MODEL_CONFIG,
        "fitted": fitted,
        "eligible_for_ensemble_audit": True,
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.component_root, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
