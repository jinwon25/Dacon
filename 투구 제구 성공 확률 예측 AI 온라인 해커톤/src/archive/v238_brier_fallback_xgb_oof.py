"""Train one strict-forward squared-error XGB on the frozen 114 features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import TARGET, YEARS, bss


PROTOCOL = "V238_BRIER_FALLBACK_XGB_STRICT_OOF_V1"
PARAMS = {
    "n_estimators": 1800,
    "learning_rate": 0.006,
    "max_depth": 10,
    # Logistic Hessians are about 0.25 around p=.5.  Scaling 6000 by four
    # preserves the incumbent's approximate minimum leaf sample support.
    "min_child_weight": 24000,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 50.0,
    "reg_alpha": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 16,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "random_state": 2028,
}


def restrictions() -> dict[str, bool]:
    return {
        "frozen_114_feature_contract_reused": True,
        "incumbent_training_rows_and_decay_reused": True,
        "single_loss_function_change": True,
        "strictly_prior_season_training": True,
        "locked_2024_not_used_for_model_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    feature_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv, usecols=["season", "game_type", TARGET], low_memory=False
    )
    features = pd.read_parquet(feature_cache)
    if len(features) != len(train) or features.shape[1] != 114:
        raise ValueError("frozen 114-feature cache mismatch")
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"hyunku_brier_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            elapsed = 0.0
            reused = True
        else:
            fit = (season < year) & ~(is_futures & (season <= 2022))
            weight = (0.5 ** ((year - 1 - season[fit]) / 2.0)).astype(np.float32)
            model = xgb.XGBRegressor(**PARAMS)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = np.clip(
                model.predict(features.loc[validation]).astype(np.float64),
                0.001,
                0.999,
            )
            elapsed = time.time() - started
            reused = False
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            print(
                f"[v238] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(((season < year) & ~(is_futures & (season <= 2022))).sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "oof_rebuilt",
        "feature_count": int(features.shape[1]),
        "feature_cache": str(feature_cache),
        "model_params": PARAMS,
        "folds": folds,
        "eligible_for_locked_replacement_audit": True,
        "eligible_for_packaging": False,
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
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.train_csv, args.feature_cache, args.output_dir),
        ensure_ascii=False, indent=2, default=float,
    ))


if __name__ == "__main__":
    main()
