"""Walk-forward experiments with fixed exponential season-decay weights."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import ID_COL, TARGET_COL, read_main
from src.followup import _upsert_csv
from src.calibration import apply_logit_offset
from src.metrics import brier_score
from src.train import train_lgb_holdout, train_rf_holdout
from src.validation import walk_forward_splits


def season_decay_weights(seasons: np.ndarray, half_life: float) -> np.ndarray:
    """Return mean-one weights whose mass halves every ``half_life`` seasons."""
    values = np.asarray(seasons, dtype=np.float64)
    if values.size == 0:
        raise ValueError("seasons cannot be empty")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    age = values.max() - values
    weights = np.exp2(-age / float(half_life))
    return weights / weights.mean()


def _incumbent_cache(project_dir: Path, year: int) -> dict[str, np.ndarray]:
    path = (
        project_dir
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def run(
    project_dir: Path,
    config_path: Path,
    validation_seasons: list[int] | None = None,
    half_lives: list[float] | None = None,
) -> pd.DataFrame:
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    followup = json.loads(
        (project_dir / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    years = validation_seasons or [int(v) for v in spec["validation_seasons"]]
    train = read_main(project_dir / "data" / "train.csv")
    test_columns = pd.read_csv(
        project_dir / "data" / "test.csv", nrows=0, encoding="utf-8-sig"
    ).columns.tolist()
    official_features = [column for column in test_columns if column != ID_COL]
    folds = {
        fold.validation_season: fold
        for fold in walk_forward_splits(train, validation_seasons=tuple(years))
    }
    incumbent_spec = followup["incumbent"]
    rows: list[dict[str, Any]] = []
    output_dir = project_dir / "artifacts" / "followup" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        fold = folds[year]
        cache = _incumbent_cache(project_dir, year)
        target = train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
        if not np.array_equal(cache["target"], target):
            raise ValueError(f"incumbent target mismatch for {year}")
        offset = float(cache["offset"][0])
        incumbent = cache["incumbent"].astype(np.float64)
        incumbent_lgb = cache["lgb_trend"].astype(np.float64)
        incumbent_rf = cache["rf_trend"].astype(np.float64)
        train_seasons = train.iloc[fold.train_idx]["season"].to_numpy()
        selected_half_lives = half_lives or [
            float(value) for value in spec["half_lives"]
        ]
        for half_life in selected_half_lives:
            print(f"[Recency training] year={year}, half_life={half_life:g}")
            weights = season_decay_weights(train_seasons, half_life)
            lgb_result = train_lgb_holdout(
                train,
                fold.train_idx,
                fold.valid_idx,
                incumbent_spec["lgb_variant"],
                int(incumbent_spec["max_boost_rounds"]),
                int(incumbent_spec["early_stopping_rounds"]),
                None,
                sample_weight=weights,
            )
            rf_result = train_rf_holdout(
                train,
                fold.train_idx,
                fold.valid_idx,
                official_features,
                random_state=int(spec["seed"]),
                sample_weight=weights,
            )
            weighted_lgb = apply_logit_offset(
                np.asarray(lgb_result["prediction"], dtype=np.float64), offset
            )
            weighted_rf = apply_logit_offset(
                np.asarray(rf_result["prediction"], dtype=np.float64), offset
            )
            candidates = {
                "weighted_lgb_only": (
                    float(incumbent_spec["lgb_weight"]) * weighted_lgb
                    + float(incumbent_spec["rf_weight"]) * incumbent_rf
                ),
                "weighted_rf_only": (
                    float(incumbent_spec["lgb_weight"]) * incumbent_lgb
                    + float(incumbent_spec["rf_weight"]) * weighted_rf
                ),
                "weighted_both": (
                    float(incumbent_spec["lgb_weight"]) * weighted_lgb
                    + float(incumbent_spec["rf_weight"]) * weighted_rf
                ),
            }
            label = str(half_life).replace(".", "p")
            np.savez_compressed(
                output_dir / f"recency_h{label}_validate_{year}.npz",
                target=target,
                valid_idx=fold.valid_idx,
                weighted_lgb=weighted_lgb,
                weighted_rf=weighted_rf,
                incumbent=incumbent,
            )
            for candidate, prediction in candidates.items():
                rows.append(
                    {
                        "outer_validation_season": year,
                        "half_life": half_life,
                        "candidate": candidate,
                        "brier": brier_score(target, prediction),
                        "incumbent_brier": brier_score(target, incumbent),
                        "delta_brier": brier_score(target, prediction)
                        - brier_score(target, incumbent),
                        "prediction_mean": float(np.mean(prediction)),
                        "target_rate": float(np.mean(target)),
                        "lgb_best_iteration": lgb_result["best_iteration"],
                        "lgb_fit_seconds": lgb_result["fit_seconds"],
                        "rf_fit_seconds": rf_result["fit_seconds"],
                        "peak_memory_mb": max(
                            lgb_result["peak_memory_mb"], rf_result["peak_memory_mb"]
                        ),
                    }
                )
            del lgb_result, rf_result
            gc.collect()
    frame = pd.DataFrame(rows)
    _upsert_csv(
        project_dir / "reports" / "recency_training_results.csv",
        frame,
        ["outer_validation_season", "half_life", "candidate"],
    )
    print(frame.to_string(index=False))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/recency_training.json")
    )
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--half-lives", nargs="+", type=float)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    run(project_dir, config_path, args.years, args.half_lives)


if __name__ == "__main__":
    main()
