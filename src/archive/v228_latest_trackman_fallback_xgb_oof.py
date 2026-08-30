"""Add latest prior-season TrackMan movement to the fallback XGB contract.

The v217 114-feature cache, XGB hyperparameters, cumulative training domain,
and recency weights are frozen.  This single challenger adds only latest
prior-season pitcher TrackMan values and their deltas from the prior-career
profile.  Every audit row uses TrackMan seasons strictly before its season.
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

from src.archive.v217_rebuild_fallback_xgb_oof import (
    PARAMS,
    TARGET,
    TM_COLUMNS,
    YEARS,
    bss,
)


PROTOCOL = "V228_LATEST_TRACKMAN_FALLBACK_XGB_STRICT_OOF_V1"
RANDOM_STATE = 2028
RAW_TM_COLUMNS = tuple(column for column in TM_COLUMNS if column != "rel_speed_sd")
LATEST_COLUMNS = tuple(f"tm_latest_{column}" for column in TM_COLUMNS)
DELTA_COLUMNS = tuple(f"tm_latest_vs_career_{column}" for column in TM_COLUMNS)
SUPPLEMENT_COLUMNS = (*LATEST_COLUMNS, *DELTA_COLUMNS, "tm_latest_age", "tm_latest_covered")


def make_season_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame
) -> pd.DataFrame:
    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(0.90), ["pitcher_id", "pitcher_trackman_id"]
    ].drop_duplicates("pitcher_trackman_id")
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    profiles = joined.groupby(["pitcher_id", "season"], sort=False)[
        list(RAW_TM_COLUMNS)
    ].mean()
    profiles["rel_speed_sd"] = joined.groupby(
        ["pitcher_id", "season"], sort=False
    )["rel_speed"].std()
    return profiles.reset_index()


def build_recent_trackman_features(
    train_context: pd.DataFrame,
    season_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build latest and prior-career profiles with a strict ``profile < row`` rule."""
    row_season = train_context["season"].to_numpy(np.int16)
    row_pitcher = train_context["pitcher_id"].to_numpy(np.int64)
    arrays = {
        column: np.full(len(train_context), np.nan, dtype=np.float32)
        for column in SUPPLEMENT_COLUMNS
    }
    for year in sorted(int(value) for value in np.unique(row_season)):
        audit = row_season == year
        prior = season_profiles.loc[season_profiles["season"].lt(year)].copy()
        if prior.empty:
            arrays["tm_latest_covered"][audit] = 0.0
            continue
        latest = (
            prior.sort_values("season")
            .groupby("pitcher_id", sort=False)
            .tail(1)
            .set_index("pitcher_id")
        )
        career = prior.groupby("pitcher_id", sort=False)[list(TM_COLUMNS)].mean()
        pitchers = pd.Series(row_pitcher[audit])
        latest_season = pitchers.map(latest["season"]).to_numpy(np.float64)
        covered = np.isfinite(latest_season)
        arrays["tm_latest_age"][audit] = (year - latest_season).astype(np.float32)
        arrays["tm_latest_covered"][audit] = covered.astype(np.float32)
        for column in TM_COLUMNS:
            latest_value = pitchers.map(latest[column]).to_numpy(np.float64)
            career_value = pitchers.map(career[column]).to_numpy(np.float64)
            arrays[f"tm_latest_{column}"][audit] = latest_value.astype(np.float32)
            arrays[f"tm_latest_vs_career_{column}"][audit] = (
                latest_value - career_value
            ).astype(np.float32)
    return pd.DataFrame(arrays, columns=SUPPLEMENT_COLUMNS)


def fit_mask(season: np.ndarray, is_futures: np.ndarray, year: int) -> np.ndarray:
    return (season < int(year)) & ~(is_futures & (season <= 2022))


def restrictions() -> dict[str, bool]:
    return {
        "base_114_features_frozen_from_v217": True,
        "xgb_hyperparameters_frozen_from_v217": True,
        "fit_domain_and_weights_frozen_from_v217": True,
        "only_latest_trackman_feature_family_added": True,
        "strictly_prior_season_trackman_profiles": True,
        "trackman_mapping_confidence_at_least_090": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    base_feature_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[TARGET, "season", "game_type", "pitcher_id"],
        low_memory=False,
    )
    base = pd.read_parquet(base_feature_cache)
    if len(base) != len(train) or base.shape[1] != 114:
        raise ValueError("v217 base feature-cache contract mismatch")
    supplement_cache = output_dir / "latest_trackman_features.parquet"
    if supplement_cache.exists():
        supplement = pd.read_parquet(supplement_cache)
        if list(supplement.columns) != list(SUPPLEMENT_COLUMNS):
            raise ValueError("cached latest TrackMan columns mismatch")
        if len(supplement) != len(train):
            raise ValueError("cached latest TrackMan rows mismatch")
        supplement_reused = True
    else:
        started = time.time()
        trackman = pd.read_csv(
            trackman_csv,
            usecols=["pitcher_trackman_id", "season", *RAW_TM_COLUMNS],
            low_memory=False,
        )
        pitcher_map = pd.read_csv(pitcher_map_csv)
        profiles = make_season_profiles(trackman, pitcher_map)
        supplement = build_recent_trackman_features(train, profiles)
        supplement.to_parquet(supplement_cache, index=False)
        supplement_reused = False
        print(
            f"[v228] built latest TrackMan rows={len(supplement):,} "
            f"cols={supplement.shape[1]} profiles={len(profiles):,} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    features = pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)], axis=1
    )

    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"hyunku_latest_tm_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = fit_mask(season, is_futures, year)
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
                f"[v228] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"coverage={float(supplement.loc[validation, 'tm_latest_covered'].mean()):.3f} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "trackman_latest_coverage": float(
                supplement.loc[validation, "tm_latest_covered"].mean()
            ),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "latest_trackman_oof_fitted",
        "base_feature_count": 114,
        "added_feature_count": len(SUPPLEMENT_COLUMNS),
        "feature_count": int(features.shape[1]),
        "added_features": list(SUPPLEMENT_COLUMNS),
        "supplement_cache": str(supplement_cache),
        "supplement_cache_reused": supplement_reused,
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
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--pitcher-map-csv", type=Path, required=True)
    parser.add_argument("--base-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.pitcher_map_csv,
        args.base_feature_cache, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
