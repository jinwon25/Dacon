"""Fit the fallback XGB only on its row-local R_CORE pressure domain.

The v217 feature cache, model parameters, random seed, and recency weights are
frozen.  The only change is the training population: prior-season Regular
R_CORE pitches with runners on or LI >= 1.5.  The probability>=0.50 rule is
still applied only by the downstream Public1175 composition audit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET, YEARS, bss


PROTOCOL = "V231_PRESSURE_SPECIALIST_FALLBACK_XGB_STRICT_OOF_V1"
RANDOM_STATE = 2028


def pressure_rcore(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    core = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    pressure = (
        frame["num_runners_on"].to_numpy(np.float64) > 0.0
    ) | (frame["li"].to_numpy(np.float64) >= 1.5)
    return core & pressure


def specialist_fit_mask(frame: pd.DataFrame, audit_year: int) -> np.ndarray:
    season = frame["season"].to_numpy(np.int16)
    return (season < int(audit_year)) & pressure_rcore(frame)


def restrictions() -> dict[str, bool]:
    return {
        "base_114_features_frozen_from_v217": True,
        "xgb_hyperparameters_frozen_from_v217": True,
        "random_seed_and_recency_weights_frozen_from_v217": True,
        "only_training_population_changed": True,
        "training_population_is_row_local_pressure_rcore": True,
        "strictly_prior_season_labels": True,
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
    columns = [
        TARGET, "season", "game_type", "pitcher_team_id", "batter_team_id",
        "num_runners_on", "li",
    ]
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    features = pd.read_parquet(feature_cache)
    if len(features) != len(train) or features.shape[1] != 114:
        raise ValueError("v217 feature-cache contract mismatch")
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"hyunku_pressure_specialist_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = specialist_fit_mask(train, year)
        if np.any(season[fit] >= year):
            raise ValueError("current/future label entered specialist fit")
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            elapsed = 0.0
            reused = True
        else:
            weight = (
                0.5 ** ((year - 1 - season[fit]) / 2.0)
            ).astype(np.float32)
            model = xgb.XGBClassifier(**PARAMS, random_state=RANDOM_STATE)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = model.predict_proba(features.loc[validation])[:, 1]
            elapsed = time.time() - started
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            reused = False
            print(
                f"[v231] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "fit_fraction_of_prior_rows": float(
                fit.sum() / max(1, np.sum(season < year))
            ),
            "audit_rows": int(validation.sum()),
            "bss_all_regular": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "pressure_specialist_oof_fitted",
        "feature_count": int(features.shape[1]),
        "feature_cache": str(feature_cache),
        "model_params": PARAMS,
        "random_state": RANDOM_STATE,
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
    result = run(args.train_csv, args.feature_cache, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
