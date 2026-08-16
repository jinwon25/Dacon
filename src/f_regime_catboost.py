"""Recent-regime F specialist with row-local CatBoost inference artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.data import TARGET_COL, read_main
from src.features import FeatureBuilder
from src.followup_models import _catboost_frame
from src.metrics import brier_score
from src.rolling_drift import build_rolling_damped_ensemble


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _catboost_params(spec: dict[str, Any], seed: int, iterations: int) -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": int(iterations),
        "depth": int(spec["depth"]),
        "learning_rate": float(spec["learning_rate"]),
        "l2_leaf_reg": float(spec["l2_leaf_reg"]),
        "random_strength": float(spec["random_strength"]),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": float(spec["bagging_temperature"]),
        "random_seed": int(seed),
        "thread_count": int(spec["thread_count"]),
        "allow_writing_files": False,
        "task_type": "CPU",
        "verbose": 100,
    }


def _indices(
    train: pd.DataFrame,
    seasons: list[int],
    game_type: str,
) -> np.ndarray:
    mask = train["season"].isin(seasons) & train["game_type"].astype("string").eq(
        game_type
    )
    return np.flatnonzero(mask.to_numpy())


def _build_pool(
    builder: FeatureBuilder,
    frame: pd.DataFrame,
    target: np.ndarray | None,
) -> tuple[pd.DataFrame, Pool, list[str]]:
    features = builder.raw_transform(frame)
    categorical = [
        column for column in builder.categorical_columns if column in features.columns
    ]
    features = _catboost_frame(features, categorical)
    pool = Pool(features, label=target, cat_features=categorical)
    return features, pool, categorical


def _trackman_reference(
    project_dir: Path,
    train: pd.DataFrame,
    validation_season: int,
    game_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    incumbent = _load_npz(
        project_dir
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{validation_season}.npz"
    )
    trackman = _load_npz(
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"lgb_trackman_pitcher_v1_validate_{validation_season}.npz"
    )
    valid_idx = incumbent["valid_idx"].astype(np.int64)
    if not np.array_equal(valid_idx, trackman["valid_idx"]):
        raise ValueError("Trackman/incumbent validation indices differ")
    fold = train.iloc[valid_idx]
    mask = fold["game_type"].astype("string").eq(game_type).to_numpy()
    season_rates = (
        train.loc[train["season"] < validation_season]
        .groupby("season", observed=True)[TARGET_COL]
        .mean()
    )
    trackman_rolling, _ = build_rolling_damped_ensemble(
        trackman["prediction"].astype(np.float64),
        season_rates,
        validation_season,
        ["damped_3_0.5", "damped_3_0.8"],
        [0.5, 0.5],
    )
    return (
        incumbent["target"].astype(np.float64)[mask],
        incumbent["incumbent"].astype(np.float64)[mask],
        trackman_rolling[mask],
    )


def validate(
    project_dir: Path,
    train: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    game_type = str(spec["game_type"])
    validation_season = int(spec["validation_season"])
    train_idx = _indices(
        train, [int(value) for value in spec["validation_train_seasons"]], game_type
    )
    valid_idx = _indices(train, [validation_season], game_type)
    y_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    target, incumbent, trackman = _trackman_reference(
        project_dir, train, validation_season, game_type
    )
    if not np.array_equal(target, y_valid):
        raise ValueError("specialist and reference targets differ")

    builder = FeatureBuilder(
        feature_set=str(spec["feature_set"]),
        drop_columns=list(spec["drop_columns"]),
    ).fit(train.iloc[train_idx], y_train)
    x_train, train_pool, _ = _build_pool(builder, train.iloc[train_idx], y_train)
    x_valid, valid_pool, _ = _build_pool(builder, train.iloc[valid_idx], y_valid)
    artifact_dir = project_dir / "artifacts" / "f_regime"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    best_iterations: list[int] = []
    for seed in [int(value) for value in spec["seeds"]]:
        started = time.perf_counter()
        model = CatBoostClassifier(
            **_catboost_params(spec, seed, int(spec["iterations"])),
            od_type="Iter",
            early_stopping_rounds=int(spec["early_stopping_rounds"]),
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
        best_iteration = int(model.get_best_iteration()) + 1
        predictions.append(prediction)
        best_iterations.append(best_iteration)
        np.savez_compressed(
            artifact_dir / f"f_catboost_seed{seed}_validate_{validation_season}.npz",
            valid_idx=valid_idx,
            target=y_valid,
            prediction=prediction,
            best_iteration=np.asarray([best_iteration], dtype=np.int32),
        )
        rows.append(
            {
                "candidate": f"f_catboost_seed{seed}",
                "validation_season": validation_season,
                "n_rows": len(y_valid),
                "brier": brier_score(y_valid, prediction),
                "prediction_mean": float(prediction.mean()),
                "best_iteration": best_iteration,
                "fit_seconds": time.perf_counter() - started,
            }
        )
        del model
        gc.collect()

    ensemble = np.mean(np.column_stack(predictions), axis=1)
    rows.append(
        {
            "candidate": "f_catboost_seed_ensemble",
            "validation_season": validation_season,
            "n_rows": len(y_valid),
            "brier": brier_score(y_valid, ensemble),
            "prediction_mean": float(ensemble.mean()),
            "best_iteration": int(round(float(np.median(best_iterations)))),
            "fit_seconds": float("nan"),
        }
    )
    for weight in (0.50, 0.65, 0.75, 0.85, 1.00):
        prediction = weight * ensemble + (1.0 - weight) * trackman
        rows.append(
            {
                "candidate": f"f_catboost_w{weight:.2f}_trackman",
                "validation_season": validation_season,
                "n_rows": len(y_valid),
                "brier": brier_score(y_valid, prediction),
                "prediction_mean": float(prediction.mean()),
                "best_iteration": int(round(float(np.median(best_iterations)))),
                "fit_seconds": float("nan"),
            }
        )
    rows.extend(
        [
            {
                "candidate": "f_incumbent_reference",
                "validation_season": validation_season,
                "n_rows": len(y_valid),
                "brier": brier_score(y_valid, incumbent),
                "prediction_mean": float(incumbent.mean()),
                "best_iteration": 0,
                "fit_seconds": 0.0,
            },
            {
                "candidate": "f_trackman_reference",
                "validation_season": validation_season,
                "n_rows": len(y_valid),
                "brier": brier_score(y_valid, trackman),
                "prediction_mean": float(trackman.mean()),
                "best_iteration": 0,
                "fit_seconds": 0.0,
            },
        ]
    )
    result = pd.DataFrame(rows).sort_values("brier", kind="stable")
    result.to_csv(
        project_dir / "reports" / "f_regime_catboost_results.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        artifact_dir / f"f_catboost_ensemble_validate_{validation_season}.npz",
        valid_idx=valid_idx,
        target=y_valid,
        prediction=ensemble,
        incumbent=incumbent,
        trackman=trackman,
        best_iterations=np.asarray(best_iterations, dtype=np.int32),
    )
    del x_train, x_valid, train_pool, valid_pool
    gc.collect()
    return result


def backtest(
    project_dir: Path,
    train: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    game_type = str(spec["game_type"])
    seed = int(spec["seeds"][0])
    iterations = int(spec["final_iterations"])
    for train_season, validation_season in spec["backtest_pairs"]:
        train_idx = _indices(train, [int(train_season)], game_type)
        valid_idx = _indices(train, [int(validation_season)], game_type)
        y_train = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
        y_valid = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
        builder = FeatureBuilder(
            feature_set=str(spec["feature_set"]),
            drop_columns=list(spec["drop_columns"]),
        ).fit(train.iloc[train_idx], y_train)
        x_train, train_pool, _ = _build_pool(builder, train.iloc[train_idx], y_train)
        x_valid, valid_pool, _ = _build_pool(builder, train.iloc[valid_idx], y_valid)
        model = CatBoostClassifier(**_catboost_params(spec, seed, iterations))
        model.fit(train_pool)
        prediction = model.predict_proba(valid_pool)[:, 1]
        rows.append(
            {
                "train_season": int(train_season),
                "validation_season": int(validation_season),
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "brier": brier_score(y_valid, prediction),
                "prediction_mean": float(prediction.mean()),
                "target_rate": float(y_valid.mean()),
                "iterations": iterations,
            }
        )
        del model, x_train, x_valid, train_pool, valid_pool
        gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(
        project_dir / "reports" / "f_regime_catboost_backtest.csv",
        index=False,
        encoding="utf-8",
    )
    return result


def train_final(
    project_dir: Path,
    train: pd.DataFrame,
    spec: dict[str, Any],
) -> dict[str, Any]:
    game_type = str(spec["game_type"])
    final_idx = _indices(
        train, [int(value) for value in spec["final_train_seasons"]], game_type
    )
    target = train.iloc[final_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder = FeatureBuilder(
        feature_set=str(spec["feature_set"]),
        drop_columns=list(spec["drop_columns"]),
    ).fit(train.iloc[final_idx], target)
    features, pool, _ = _build_pool(builder, train.iloc[final_idx], target)
    artifact_dir = project_dir / "artifacts" / "f_regime" / "final"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    builder.save_spec(artifact_dir / "f_catboost_feature_spec.json")
    model_names: list[str] = []
    model_sizes: list[int] = []
    final_seeds = [
        int(value) for value in spec.get("final_seeds", spec["seeds"])
    ]
    for seed in final_seeds:
        model = CatBoostClassifier(
            **_catboost_params(spec, seed, int(spec["final_iterations"]))
        )
        model.fit(pool)
        name = f"f_catboost_seed{seed}.cbm"
        model.save_model(str(artifact_dir / name))
        model_names.append(name)
        model_sizes.append((artifact_dir / name).stat().st_size)
        del model
        gc.collect()
    manifest = {
        "candidate": "f_recent_regime_catboost_seed_ensemble_v1",
        "game_type": game_type,
        "train_seasons": [int(value) for value in spec["final_train_seasons"]],
        "n_rows": len(final_idx),
        "target_rate": float(target.mean()),
        "model_names": model_names,
        "model_sizes": model_sizes,
        "iterations": int(spec["final_iterations"]),
        "seeds": final_seeds,
        "f_catboost_weight": float(spec["f_catboost_weight"]),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del features, pool
    gc.collect()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/f_regime_catboost.json")
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["validate", "backtest", "final"],
        default=["validate", "backtest", "final"],
    )
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    train = read_main(project_dir / "data" / "train.csv")
    if "validate" in args.stages:
        print(validate(project_dir, train, spec).to_string(index=False), flush=True)
    if "backtest" in args.stages:
        print(backtest(project_dir, train, spec).to_string(index=False), flush=True)
    if "final" in args.stages:
        print(json.dumps(train_final(project_dir, train, spec), indent=2), flush=True)


if __name__ == "__main__":
    main()
