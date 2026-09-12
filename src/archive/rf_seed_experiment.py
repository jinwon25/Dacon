"""Fixed three-seed probability average for the incumbent RandomForest."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset
from src.archive.data import ID_COL, TARGET_COL, read_main
from src.archive.domain_drift import (
    apply_game_type_offsets,
    fit_game_type_regime_offsets,
)
from src.archive.followup import _load_all_caches
from src.metrics import brier_score, cluster_bootstrap_delta
from src.archive.train import train_rf_holdout


SEEDS = (42, 202, 777)


def _cache_path(project_dir: Path, seed: int, year: int) -> Path:
    return (
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"rf_official_seed{seed}_validate_{year}.npz"
    )


def _prediction_for_seed(
    project_dir: Path,
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    features: list[str],
    seed: int,
    year: int,
) -> np.ndarray:
    path = _cache_path(project_dir, seed, year)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["prediction"].astype(np.float64)
    result = train_rf_holdout(
        train,
        train_idx,
        valid_idx,
        features,
        random_state=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prediction=np.asarray(result["prediction"], dtype=np.float64),
        fit_seconds=np.array([result["fit_seconds"]], dtype=np.float64),
        inference_seconds=np.array([result["inference_seconds"]], dtype=np.float64),
        peak_memory_mb=np.array([result["peak_memory_mb"]], dtype=np.float64),
        random_state=np.array([seed], dtype=np.int32),
    )
    del result
    gc.collect()
    with np.load(path, allow_pickle=False) as saved:
        return saved["prediction"].astype(np.float64)


def run(project_dir: Path, years: list[int]) -> pd.DataFrame:
    config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    train = read_main(project_dir / "data" / "train.csv")
    features = [
        column
        for column in read_main(project_dir / "data" / "test.csv", nrows=0).columns
        if column != ID_COL
    ]
    folds, caches = _load_all_caches(project_dir, config, train)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    prediction_store: dict[str, dict[int, np.ndarray]] = {
        "rf_seed_average": {},
        "rf_seed_average_plus_regime": {},
    }

    for year in years:
        fold = folds[year]
        cache = caches[year]
        rf_predictions = [cache["rf_raw"]]
        for seed in SEEDS[1:]:
            print(f"training/loading RF seed={seed}, validation={year}")
            rf_predictions.append(
                _prediction_for_seed(
                    project_dir,
                    train,
                    fold.train_idx,
                    fold.valid_idx,
                    features,
                    seed,
                    year,
                )
            )
        rf_average = np.mean(rf_predictions, axis=0)
        offset = float(cache["offset"])
        rf_trend = apply_logit_offset(rf_average, offset)
        seed_average = 0.35 * cache["lgb_trend"] + 0.65 * rf_trend
        offsets, _ = fit_game_type_regime_offsets(
            train.iloc[fold.train_idx], year
        )
        seed_plus_regime = apply_game_type_offsets(
            seed_average,
            train.iloc[fold.valid_idx]["game_type"],
            offsets,
        )
        predictions = {
            "rf_seed_average": seed_average,
            "rf_seed_average_plus_regime": seed_plus_regime,
        }
        target = cache["target"]
        incumbent = cache["incumbent"]
        valid = train.iloc[fold.valid_idx]
        for candidate, prediction in predictions.items():
            prediction_store[candidate][year] = prediction
            rows.append(
                {
                    "candidate": candidate,
                    "outer_validation_season": year,
                    "brier": brier_score(target, prediction),
                    "incumbent_brier": brier_score(target, incumbent),
                    "delta_brier": brier_score(target, prediction)
                    - brier_score(target, incumbent),
                    "prediction_mean": float(prediction.mean()),
                    "target_rate": float(target.mean()),
                    "seeds": ";".join(str(value) for value in SEEDS),
                }
            )
            clusters = (
                valid["pitcher_id"].astype("string")
                + "-"
                + valid["season"].astype("string")
            )
            bootstrap_rows.append(
                {
                    "scope": str(year),
                    "candidate": candidate,
                    "cluster_type": "pitcher_season",
                    **cluster_bootstrap_delta(
                        target,
                        prediction,
                        incumbent,
                        clusters,
                        n_resamples=2_000,
                        seed=40_000 + year,
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(project_dir / "research" / "reports" / "rf_seed_results.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        project_dir / "research" / "reports" / "rf_seed_bootstrap.csv", index=False
    )
    print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()
    run(args.project_dir.resolve(), args.years)


if __name__ == "__main__":
    main()
