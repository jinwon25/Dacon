"""Wave 2-4 hierarchical, CatBoost, and Brier-objective experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL, read_main
from src.archive.experiment_tracking import append_experiment
from src.archive.followup import _load_all_caches, _metric_row, _upsert_csv
from src.archive.followup_models import (
    train_catboost_holdout,
    train_rf_variant_holdout,
    train_xgboost_holdout,
)
from src.archive.hierarchical import HierarchicalBackoff
from src.metrics import brier_score, brier_skill_score
from src.archive.train import train_lgb_holdout


def _model_cache_path(project_dir: Path, model_name: str, year: int) -> Path:
    return (
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"{model_name}_validate_{year}.npz"
    )


def _save_model_result(
    path: Path,
    result: dict[str, Any],
    valid_idx: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        valid_idx=valid_idx.astype(np.int64),
        target=np.asarray(result["target"], dtype=np.int8),
        prediction=np.asarray(result["prediction"], dtype=np.float64),
        best_iteration=np.array([result["best_iteration"]], dtype=np.int32),
        build_seconds=np.array([result["build_seconds"]], dtype=np.float64),
        fit_seconds=np.array([result["fit_seconds"]], dtype=np.float64),
        inference_seconds=np.array([result["inference_seconds"]], dtype=np.float64),
        model_size_mb=np.array([result["model_size_mb"]], dtype=np.float64),
        peak_memory_mb=np.array([result["peak_memory_mb"]], dtype=np.float64),
    )


def _load_model_result(path: Path, valid_idx: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    if not np.array_equal(values["valid_idx"], valid_idx):
        raise ValueError(f"model cache indices mismatch: {path}")
    if not np.array_equal(values["target"], target):
        raise ValueError(f"model cache target mismatch: {path}")
    return values


def _run_model(
    project_dir: Path,
    train: pd.DataFrame,
    fold,
    config: dict[str, Any],
) -> dict[str, Any]:
    name = config["name"]
    year = fold.validation_season
    path = _model_cache_path(project_dir, name, year)
    target = train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    if path.exists():
        print(f"[Wave 2-4] Loading {name} {year} cache")
        return _load_model_result(path, fold.valid_idx, target)
    print(f"[Wave 2-4] Training {name} for validation {year}")
    if config["library"] == "lightgbm":
        result = train_lgb_holdout(
            train,
            fold.train_idx,
            fold.valid_idx,
            config,
            int(config["max_boost_rounds"]),
            int(config["early_stopping_rounds"]),
            None,
        )
    elif config["library"] == "catboost":
        result = train_catboost_holdout(train, fold.train_idx, fold.valid_idx, config)
    elif config["library"] == "xgboost":
        result = train_xgboost_holdout(train, fold.train_idx, fold.valid_idx, config)
    elif config["library"] == "random_forest":
        result = train_rf_variant_holdout(train, fold.train_idx, fold.valid_idx, config)
    else:
        raise ValueError(f"unknown library: {config['library']}")
    _save_model_result(path, result, fold.valid_idx)
    del result
    gc.collect()
    return _load_model_result(path, fold.valid_idx, target)


def run_hierarchical(
    project_dir: Path,
    train: pd.DataFrame,
    folds: dict[int, Any],
    incumbent_caches: dict[int, dict[str, Any]],
    alpha_pairs: list[list[float]],
    years: list[int],
) -> None:
    rows = []
    diversity_rows = []
    for alpha_leaf, alpha_parent in alpha_pairs:
        name = f"hier_backoff_a{int(alpha_leaf)}_{int(alpha_parent)}"
        for year in years:
            fold = folds[year]
            path = _model_cache_path(project_dir, name, year)
            target = train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
            if path.exists():
                cached = _load_model_result(path, fold.valid_idx, target)
                prediction = cached["prediction"]
            else:
                print(f"[Wave 2] Fitting {name} for validation {year}")
                encoder = HierarchicalBackoff(alpha_leaf, alpha_parent).fit(
                    train.iloc[fold.train_idx]
                )
                prediction = encoder.predict(train.iloc[fold.valid_idx])
                result = {
                    "target": target,
                    "prediction": prediction,
                    "best_iteration": 0,
                    "build_seconds": 0.0,
                    "fit_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "model_size_mb": 0.0,
                    "peak_memory_mb": 0.0,
                }
                _save_model_result(path, result, fold.valid_idx)
                del encoder
                gc.collect()
            incumbent = incumbent_caches[year]["incumbent"]
            rows.append(
                _metric_row(
                    f"W2_{name}_{year}",
                    year,
                    name,
                    prediction,
                    target,
                    incumbent,
                    diagnostic=False,
                    prediction_source="outer-train labels only; hierarchical context backoff",
                )
            )
            diversity_rows.append(
                {
                    "experiment_id": f"W2_{name}_{year}",
                    "outer_validation_season": year,
                    "model": name,
                    "correlation_incumbent": float(np.corrcoef(prediction, incumbent)[0, 1]),
                    "correlation_lgb": float(
                        np.corrcoef(prediction, incumbent_caches[year]["lgb_trend"])[0, 1]
                    ),
                    "correlation_rf": float(
                        np.corrcoef(prediction, incumbent_caches[year]["rf_trend"])[0, 1]
                    ),
                }
            )
    _upsert_csv(
        project_dir / "research" / "reports" / "walk_forward_results.csv",
        pd.DataFrame(rows),
        ["experiment_id"],
    )
    _upsert_csv(
        project_dir / "research" / "reports" / "model_diversity.csv",
        pd.DataFrame(diversity_rows),
        ["experiment_id"],
    )


def run_models(
    project_dir: Path,
    train: pd.DataFrame,
    folds: dict[int, Any],
    incumbent_caches: dict[int, dict[str, Any]],
    model_configs: dict[str, dict[str, Any]],
    model_names: list[str],
    years: list[int],
) -> None:
    rows = []
    diversity_rows = []
    experiments_path = project_dir / "research" / "reports" / "experiments.csv"
    existing_ids = (
        set(pd.read_csv(experiments_path, usecols=["experiment_id"])["experiment_id"].astype(str))
        if experiments_path.exists()
        else set()
    )
    for name in model_names:
        config = model_configs[name]
        for year in years:
            fold = folds[year]
            cached = _run_model(project_dir, train, fold, config)
            target = cached["target"]
            prediction = cached["prediction"]
            incumbent = incumbent_caches[year]["incumbent"]
            wave = 3 if config["library"] == "catboost" else 4
            experiment_id = f"W{wave}_{name}_{year}"
            rows.append(
                _metric_row(
                    experiment_id,
                    year,
                    name,
                    prediction,
                    target,
                    incumbent,
                    diagnostic=False,
                    prediction_source=f"strict outer train; {config['library']}; season dropped",
                )
            )
            diversity_rows.append(
                {
                    "experiment_id": experiment_id,
                    "outer_validation_season": year,
                    "model": name,
                    "correlation_incumbent": float(np.corrcoef(prediction, incumbent)[0, 1]),
                    "correlation_lgb": float(
                        np.corrcoef(prediction, incumbent_caches[year]["lgb_trend"])[0, 1]
                    ),
                    "correlation_rf": float(
                        np.corrcoef(prediction, incumbent_caches[year]["rf_trend"])[0, 1]
                    ),
                }
            )
            if experiment_id not in existing_ids:
                append_experiment(
                    experiments_path,
                    {
                        "experiment_id": experiment_id,
                        "validation_split": f"train_before_{year}_validate_{year}",
                        "feature_groups": (
                            f"{config.get('feature_set', 'official_features')}; season dropped"
                        ),
                        "model_params": json.dumps(config, sort_keys=True),
                        "seed": config.get("seed", 42),
                        "raw_brier_score": f"{brier_score(target, prediction):.9f}",
                        "local_brier_skill_score": f"{brier_skill_score(target, prediction):.3f}",
                        "calibration": "none",
                        "training_seconds": f"{float(cached['fit_seconds'][0] + cached['build_seconds'][0]):.3f}",
                        "inference_seconds": f"{float(cached['inference_seconds'][0]):.3f}",
                        "model_size_mb": f"{float(cached['model_size_mb'][0]):.3f}",
                        "peak_memory_mb": f"{float(cached['peak_memory_mb'][0]):.1f}",
                        "leakage_risk_notes": "outer-train category map/ordered statistics only; season, row_id and CSV order excluded",
                    },
                )
                existing_ids.add(experiment_id)
            print(
                f"[Wave 2-4] {name} {year}: {brier_score(target, prediction):.9f} "
                f"delta={brier_score(target, prediction)-brier_score(target, incumbent):+.9f}"
            )
    _upsert_csv(
        project_dir / "research" / "reports" / "walk_forward_results.csv",
        pd.DataFrame(rows),
        ["experiment_id"],
    )
    _upsert_csv(
        project_dir / "research" / "reports" / "model_diversity.csv",
        pd.DataFrame(diversity_rows),
        ["experiment_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("research/configs/wave234.json"))
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train = read_main(project_dir / "data" / "train.csv")
    followup_config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    folds, incumbent_caches = _load_all_caches(project_dir, followup_config, train)
    years = [int(year) for year in args.years]
    if args.hierarchical:
        run_hierarchical(
            project_dir,
            train,
            folds,
            incumbent_caches,
            config["hierarchical_alpha_pairs"],
            years,
        )
    if args.models:
        unknown = sorted(set(args.models) - set(config["model_configs"]))
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
        run_models(
            project_dir,
            train,
            folds,
            incumbent_caches,
            config["model_configs"],
            args.models,
            years,
        )


if __name__ == "__main__":
    main()
