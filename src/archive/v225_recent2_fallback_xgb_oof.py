"""Fit the frozen Public1175 fallback XGB on only two prior seasons.

This is a single, pre-registered adaptation test.  The 114-feature contract,
XGB hyperparameters, random seed, row-domain exclusion, and recency weights
are inherited from v217.  Only the lower bound of the training window changes.
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


PROTOCOL = "V225_RECENT2_FALLBACK_XGB_STRICT_OOF_V1"
WINDOW_SEASONS = 2
RANDOM_STATE = 2028


def recent_fit_mask(
    season: np.ndarray,
    is_futures: np.ndarray,
    audit_year: int,
) -> np.ndarray:
    """Return the v217 fit domain intersected with two prior seasons."""
    season = np.asarray(season)
    is_futures = np.asarray(is_futures, dtype=bool)
    if season.shape != is_futures.shape:
        raise ValueError("season/is_futures shape mismatch")
    return (
        (season >= int(audit_year) - WINDOW_SEASONS)
        & (season < int(audit_year))
        & ~(is_futures & (season <= 2022))
    )


def recency_weight(season: np.ndarray, audit_year: int) -> np.ndarray:
    """Keep the original v217 half-life weighting unchanged."""
    season = np.asarray(season, dtype=np.float64)
    return (0.5 ** ((int(audit_year) - 1 - season) / 2.0)).astype(np.float32)


def restrictions() -> dict[str, bool]:
    return {
        "feature_contract_frozen_from_v217": True,
        "xgb_hyperparameters_frozen_from_v217": True,
        "random_seed_frozen_from_v217": True,
        "recency_weights_frozen_from_v217": True,
        "only_training_window_changed": True,
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
    train = pd.read_csv(
        train_csv,
        usecols=[TARGET, "season", "game_type"],
        low_memory=False,
    )
    features = pd.read_parquet(feature_cache)
    if len(features) != len(train):
        raise ValueError(
            f"feature-cache row mismatch: features={len(features)} train={len(train)}"
        )
    if features.shape[1] != 114:
        raise ValueError(f"feature-cache column mismatch: {features.shape[1]}")

    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"hyunku_recent2_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = recent_fit_mask(season, is_futures, year)
        fit_seasons = sorted(int(value) for value in np.unique(season[fit]))
        expected = [year - 2, year - 1]
        if fit_seasons != expected:
            raise ValueError(
                f"recent-two fit seasons mismatch: year={year} "
                f"expected={expected} actual={fit_seasons}"
            )
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            elapsed = 0.0
            reused = True
        else:
            model = xgb.XGBClassifier(**PARAMS, random_state=RANDOM_STATE)
            weight = recency_weight(season[fit], year)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = model.predict_proba(features.loc[validation])[:, 1]
            elapsed = time.time() - started
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            reused = False
            print(
                f"[v225] fold={year} seasons={fit_seasons} "
                f"fit={int(fit.sum()):,} audit={int(validation.sum()):,} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_seasons": fit_seasons,
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "recent2_oof_fitted",
        "single_preregistered_window_seasons": WINDOW_SEASONS,
        "feature_cache": str(feature_cache),
        "feature_count": int(features.shape[1]),
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
