"""Re-audit TrackMan experts with the exact frozen deployment transform.

The v252/v258 models used strict prior-season TrackMan supplements, but their
audit rows still used v217's historical training transform for the 114 base
features.  The standalone release uses ``fallback_xgb_frozen_runtime.build``.
This module keeps the original fit rows, weights, hyperparameters, and
supplements fixed while changing only audit-row base features to that runtime
transform.  Fold models are saved so the contract can be inspected later.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fallback_xgb import fallback_xgb_frozen_runtime as runtime
from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET, bss


PROTOCOL = "V265_RUNTIME_FAITHFUL_TRACKMAN_EXPERT_OOF_V1"
EXPERTS = {
    "command": {
        "prediction_stem": "command_dispersion_xgb",
        "random_state": 2052,
    },
    "batter": {
        "prediction_stem": "batter_trackman_xgb",
        "random_state": 2058,
    },
}


def combine_features(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(supplement):
        raise ValueError("base and supplement row counts differ")
    overlap = set(base).intersection(supplement)
    if overlap:
        raise ValueError(f"feature columns overlap: {sorted(overlap)[:3]}")
    return pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)],
        axis=1,
        copy=False,
    )


def fold_masks(
    season: np.ndarray, is_futures: np.ndarray, year: int
) -> tuple[np.ndarray, np.ndarray]:
    validation = (season == int(year)) & ~is_futures
    fit = (season < int(year)) & ~(is_futures & (season <= 2022))
    return fit, validation


def run(
    train_csv: Path,
    base_train_cache: Path,
    supplement_cache: Path,
    runtime_lookup_dir: Path,
    output_dir: Path,
    expert: str,
    years: tuple[int, ...],
    n_jobs: int,
) -> dict[str, Any]:
    if expert not in EXPERTS:
        raise ValueError(f"unknown expert: {expert}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    base_train = pd.read_parquet(base_train_cache)
    supplement = pd.read_parquet(supplement_cache)
    if not (len(train) == len(base_train) == len(supplement)):
        raise ValueError("train feature-cache row mismatch")

    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    config = EXPERTS[expert]
    params = dict(PARAMS)
    params["n_jobs"] = int(n_jobs)
    feature_columns = [*base_train.columns, *supplement.columns]
    folds: dict[str, Any] = {}

    for year in years:
        prediction_path = output_dir / f"{config['prediction_stem']}_{year}.npy"
        model_path = output_dir / f"{config['prediction_stem']}_{year}.json"
        fit, validation = fold_masks(season, is_futures, year)
        if prediction_path.exists():
            prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            elapsed = 0.0
            reused = True
        else:
            lookup = runtime_lookup_dir / f"lookup_{year}"
            if not (lookup / "fallback_lookups.joblib").exists():
                raise FileNotFoundError(f"missing runtime lookup: {lookup}")
            started = time.time()
            audit_base = runtime.build(train.loc[validation].reset_index(drop=True), lookup)
            if list(audit_base.columns) != list(base_train.columns):
                raise ValueError(f"runtime base feature-column mismatch: {year}")
            training_features = combine_features(
                base_train.loc[fit], supplement.loc[fit]
            ).reindex(columns=feature_columns)
            audit_features = combine_features(
                audit_base, supplement.loc[validation]
            ).reindex(columns=feature_columns)
            weight = (
                0.5 ** ((year - 1 - season[fit]) / 2.0)
            ).astype(np.float32)
            model = xgb.XGBClassifier(
                **params, random_state=int(config["random_state"])
            )
            model.fit(training_features, target[fit], sample_weight=weight)
            prediction = model.predict_proba(audit_features)[:, 1].astype(np.float64)
            model.save_model(model_path)
            np.save(prediction_path, prediction.astype(np.float32), allow_pickle=False)
            elapsed = time.time() - started
            reused = False
            del model, training_features, audit_features, audit_base
            gc.collect()
            print(
                f"[v265] expert={expert} year={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} bss={bss(target[validation], prediction):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "prediction": str(prediction_path),
            "model": str(model_path),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "runtime_faithful_expert_oof_built",
        "expert": expert,
        "years": list(years),
        "base_feature_count": int(base_train.shape[1]),
        "supplement_feature_count": int(supplement.shape[1]),
        "feature_count": int(len(feature_columns)),
        "model_params": params,
        "folds": folds,
        "restrictions": {
            "original_fit_rows_weights_hyperparameters_frozen": True,
            "audit_base_uses_release_runtime_transform": True,
            "strictly_prior_season_target_free_trackman_supplement": True,
            "test_csv_read": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--base-train-cache", type=Path, required=True)
    parser.add_argument("--supplement-cache", type=Path, required=True)
    parser.add_argument("--runtime-lookup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert", choices=sorted(EXPERTS), required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_train_cache,
        args.supplement_cache,
        args.runtime_lookup_dir,
        args.output_dir,
        args.expert,
        tuple(args.years),
        args.n_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
