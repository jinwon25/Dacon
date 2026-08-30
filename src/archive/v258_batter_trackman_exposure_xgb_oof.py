"""Strict-forward batter TrackMan exposure-profile XGB trial.

Most prior TrackMan work in this project described pitchers.  This independent
feature family instead summarizes the pitch shapes, arsenals, count pressure,
and opponent variety a batter faced in prior seasons, using the target-free
sequence-derived batter identity map.  At inference it joins only on the
current row's batter ID and never reads or aggregates other test rows.
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


PROTOCOL = "V258_BATTER_TRACKMAN_EXPOSURE_XGB_STRICT_FORWARD_V1"
RANDOM_STATE = 2058
MINIMUM_MAP_DOMINANCE = 0.99
MINIMUM_MAP_SUPPORT = 20
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)


def make_batter_profiles(
    trackman: pd.DataFrame, batter_map: pd.DataFrame
) -> pd.DataFrame:
    """Return one target-free exposure profile per mapped batter-season."""

    mapping = batter_map.loc[
        batter_map["dominance"].ge(MINIMUM_MAP_DOMINANCE)
        & batter_map["support"].ge(MINIMUM_MAP_SUPPORT),
        ["batter_id", "batter_trackman_id", "support", "dominance"],
    ]
    mapping = (
        mapping.sort_values(["dominance", "support"])
        .drop_duplicates("batter_trackman_id", keep="last")
        .drop(columns=["support", "dominance"])
    )
    joined = trackman.merge(mapping, on="batter_trackman_id", how="inner")
    for metric in METRICS:
        joined[metric] = pd.to_numeric(joined[metric], errors="coerce")
    balls = pd.to_numeric(joined["balls_before"], errors="coerce")
    strikes = pd.to_numeric(joined["strikes_before"], errors="coerce")
    joined["three_ball"] = balls.eq(3).astype(np.float32)
    joined["two_strike"] = strikes.eq(2).astype(np.float32)
    joined["hitter_ahead"] = balls.gt(strikes).astype(np.float32)
    joined["pitcher_ahead"] = strikes.gt(balls).astype(np.float32)
    joined["left_pitcher"] = (
        joined["pitcher_hand"].astype(str).str.lower().str.startswith("l")
    ).astype(np.float32)

    keys = ["batter_id", "season"]
    grouped = joined.groupby(keys, observed=True, sort=False)
    profile = grouped.size().rename("pitch_count").to_frame()
    profile["game_count"] = grouped["trackman_game_id"].nunique()
    profile["pitcher_count"] = grouped["pitcher_trackman_id"].nunique()
    for column in (
        "three_ball",
        "two_strike",
        "hitter_ahead",
        "pitcher_ahead",
        "left_pitcher",
    ):
        profile[f"share_{column}"] = grouped[column].mean()
    moments = grouped[list(METRICS)].agg(["mean", "std"])
    moments.columns = [f"{stat}_{metric}" for metric, stat in moments.columns]
    profile = profile.join(moments)
    pitch_counts = (
        joined.groupby([*keys, "pitch_type_group"], observed=True, sort=False)
        .size()
        .unstack("pitch_type_group", fill_value=0)
    )
    total = pitch_counts.sum(axis=1).replace(0.0, np.nan)
    shares = []
    for group in PITCH_GROUPS:
        count = pitch_counts[group] if group in pitch_counts else 0.0
        profile[f"share_pitch_{group}"] = count / total
        shares.append(profile[f"share_pitch_{group}"].fillna(0.0))
    entropy = sum(
        -(share * np.log(share.clip(lower=1e-12))) for share in shares
    )
    profile["pitch_type_entropy"] = entropy
    profile["log_pitch_count"] = np.log1p(profile["pitch_count"])
    profile["log_game_count"] = np.log1p(profile["game_count"])
    profile["log_pitcher_count"] = np.log1p(profile["pitcher_count"])
    profile = profile.drop(columns=["pitch_count", "game_count", "pitcher_count"])
    return profile.reset_index()


def profile_feature_columns(profiles: pd.DataFrame) -> list[str]:
    return [column for column in profiles if column not in {"batter_id", "season"}]


def build_batter_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Attach latest, prior-career, and latest-minus-career batter profiles."""

    profile_columns = profile_feature_columns(profiles)
    output_columns = [
        name
        for column in profile_columns
        for name in (
            f"bat_tm_latest_{column}",
            f"bat_tm_career_{column}",
            f"bat_tm_latest_vs_career_{column}",
        )
    ] + ["bat_tm_profile_age", "bat_tm_profile_covered"]
    arrays = {
        column: np.full(len(rows), np.nan, dtype=np.float32)
        for column in output_columns
    }
    row_season = rows["season"].to_numpy(np.int16)
    row_batter = rows["batter_id"].to_numpy(np.int64)
    for year in sorted(int(value) for value in np.unique(row_season)):
        audit = row_season == year
        prior = profiles.loc[profiles["season"].lt(year)]
        if prior.empty:
            arrays["bat_tm_profile_covered"][audit] = 0.0
            continue
        latest = (
            prior.sort_values("season")
            .groupby("batter_id", sort=False)
            .tail(1)
            .set_index("batter_id")
        )
        career = prior.groupby("batter_id", sort=False)[profile_columns].mean()
        batters = pd.Series(row_batter[audit])
        latest_season = batters.map(latest["season"]).to_numpy(np.float64)
        arrays["bat_tm_profile_age"][audit] = year - latest_season
        arrays["bat_tm_profile_covered"][audit] = np.isfinite(
            latest_season
        ).astype(np.float32)
        for column in profile_columns:
            latest_value = batters.map(latest[column]).to_numpy(np.float64)
            career_value = batters.map(career[column]).to_numpy(np.float64)
            arrays[f"bat_tm_latest_{column}"][audit] = latest_value
            arrays[f"bat_tm_career_{column}"][audit] = career_value
            arrays[f"bat_tm_latest_vs_career_{column}"][audit] = (
                latest_value - career_value
            )
    return pd.DataFrame(arrays, columns=output_columns)


def fit_mask(season: np.ndarray, is_futures: np.ndarray, year: int) -> np.ndarray:
    return (season < int(year)) & ~(is_futures & (season <= 2022))


def run(
    train_csv: Path,
    trackman_csv: Path,
    batter_map_csv: Path,
    base_feature_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[TARGET, "season", "game_type", "batter_id"],
        low_memory=False,
    )
    base = pd.read_parquet(base_feature_cache)
    if len(base) != len(train) or base.shape[1] != 114:
        raise ValueError("v217 base feature-cache contract mismatch")
    supplement_cache = output_dir / "batter_trackman_features.parquet"
    profile_cache = output_dir / "batter_trackman_profiles.parquet"
    if supplement_cache.exists() and profile_cache.exists():
        supplement = pd.read_parquet(supplement_cache)
        profiles = pd.read_parquet(profile_cache)
        cache_reused = True
    else:
        started = time.time()
        trackman = pd.read_csv(
            trackman_csv,
            usecols=[
                "batter_trackman_id",
                "pitcher_trackman_id",
                "season",
                "trackman_game_id",
                "balls_before",
                "strikes_before",
                "pitcher_hand",
                "pitch_type_group",
                *METRICS,
            ],
            low_memory=False,
        )
        batter_map = pd.read_csv(batter_map_csv)
        batter_map.to_csv(output_dir / "batter_map_snapshot.csv", index=False)
        profiles = make_batter_profiles(trackman, batter_map)
        supplement = build_batter_features(train, profiles)
        profiles.to_parquet(profile_cache, index=False)
        supplement.to_parquet(supplement_cache, index=False)
        cache_reused = False
        print(
            f"[v258] profiles={len(profiles):,} supplement={supplement.shape} "
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
        checkpoint = output_dir / f"batter_trackman_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = fit_mask(season, is_futures, year)
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            elapsed = 0.0
            reused = True
        else:
            weight = (0.5 ** ((year - 1 - season[fit]) / 2.0)).astype(np.float32)
            model = xgb.XGBClassifier(**PARAMS, random_state=RANDOM_STATE)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = model.predict_proba(features.loc[validation])[:, 1]
            elapsed = time.time() - started
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            reused = False
            print(
                f"[v258] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"coverage={float(supplement.loc[validation, 'bat_tm_profile_covered'].mean()):.3f} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if len(prediction) != int(validation.sum()):
            raise ValueError(f"checkpoint length mismatch: {year}")
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "coverage": float(
                supplement.loc[validation, "bat_tm_profile_covered"].mean()
            ),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "strict_forward_oof_fitted",
        "family_trial_count": 1,
        "base_feature_count": 114,
        "added_feature_count": int(supplement.shape[1]),
        "feature_count": int(features.shape[1]),
        "profile_rows": int(len(profiles)),
        "batter_map_snapshot": str(output_dir / "batter_map_snapshot.csv"),
        "cache_reused": cache_reused,
        "folds": folds,
        "eligible_for_locked_route_audit": True,
        "eligible_for_packaging": False,
        "restrictions": {
            "v217_base_hyperparameters_domain_and_weights_frozen": True,
            "target_free_sequence_identity_map_only": True,
            "strictly_prior_season_trackman_profiles": True,
            "current_row_batter_id_only": True,
            "single_preregistered_feature_family": True,
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
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--batter-map-csv", type=Path, required=True)
    parser.add_argument("--base-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.batter_map_csv,
        args.base_feature_cache,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
