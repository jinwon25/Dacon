"""Strict-forward count-conditioned TrackMan arsenal/command XGB trial.

Unlike the prior unconditional pitcher TrackMan profiles, this family asks a
baseball-specific question: does a pitcher's prior-season arsenal and release
repeatability *in the current count* explain command beyond the official ASOF
rates?  It uses only the current row's count and target-free TrackMan history
from seasons strictly before the predicted row.
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


PROTOCOL = "V257_COUNT_CONDITIONED_TRACKMAN_XGB_STRICT_FORWARD_V1"
RANDOM_STATE = 2057
GLOBAL_COUNT_STATE = -1
RELIABILITY_PRIOR_PITCHES = 64.0
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
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


def _aggregate_profile(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True, sort=False)
    profile = grouped.size().rename("pitch_count").to_frame()
    moments = grouped[list(METRICS)].agg(["mean", "std"])
    moments.columns = [f"{stat}_{metric}" for metric, stat in moments.columns]
    profile = profile.join(moments)
    pitch_counts = (
        frame.loc[frame["pitch_type_group"].isin(PITCH_GROUPS)]
        .groupby([*keys, "pitch_type_group"], observed=True, sort=False)
        .size()
        .unstack("pitch_type_group", fill_value=0)
    )
    total = pitch_counts.sum(axis=1).replace(0.0, np.nan)
    for group in PITCH_GROUPS:
        count = pitch_counts[group] if group in pitch_counts else 0.0
        profile[f"share_{group}"] = count / total
    return profile.reset_index()


def make_count_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame
) -> pd.DataFrame:
    """Build exact-count and all-count summaries for each pitcher-season."""

    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(0.90), ["pitcher_id", "pitcher_trackman_id"]
    ].drop_duplicates("pitcher_trackman_id")
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    joined["count_state"] = (
        pd.to_numeric(joined["balls_before"], errors="coerce") * 3
        + pd.to_numeric(joined["strikes_before"], errors="coerce")
    )
    joined = joined.loc[joined["count_state"].between(0, 11)].copy()
    joined["count_state"] = joined["count_state"].astype(np.int8)
    for metric in METRICS:
        joined[metric] = pd.to_numeric(joined[metric], errors="coerce")
    exact = _aggregate_profile(
        joined, ["pitcher_id", "season", "count_state"]
    )
    global_profile = _aggregate_profile(joined, ["pitcher_id", "season"])
    global_profile["count_state"] = GLOBAL_COUNT_STATE
    columns = ["pitcher_id", "season", "count_state"] + [
        column
        for column in exact.columns
        if column not in {"pitcher_id", "season", "count_state"}
    ]
    return pd.concat(
        [exact[columns], global_profile[columns]], ignore_index=True
    )


def _lookup(frame: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    indexed = frame.set_index(["pitcher_id", "count_state"])
    if indexed.index.has_duplicates:
        raise ValueError("profile lookup is not unique")
    return indexed.reindex(keys)


def feature_names() -> list[str]:
    names = [
        "tm_count_profile_age",
        "tm_count_profile_covered",
        "tm_count_exact_covered",
        "tm_count_pitch_count",
        "tm_count_log_pitch_count",
        "tm_count_reliability",
    ]
    for group in PITCH_GROUPS:
        names.extend(
            [
                f"tm_count_share_{group}",
                f"tm_global_share_{group}",
                f"tm_count_share_delta_{group}",
            ]
        )
    for metric in METRICS:
        names.extend(
            [
                f"tm_count_mean_{metric}",
                f"tm_global_mean_{metric}",
                f"tm_count_reliable_mean_delta_{metric}",
                f"tm_count_sd_{metric}",
                f"tm_global_sd_{metric}",
                f"tm_count_reliable_sd_ratio_{metric}",
            ]
        )
    return names


def build_count_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Attach latest strictly-prior exact-count profiles with global backoff."""

    names = feature_names()
    arrays = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in names}
    row_season = rows["season"].to_numpy(np.int16)
    row_pitcher = rows["pitcher_id"].to_numpy(np.int64)
    row_count = (
        rows["balls_before"].to_numpy(np.int16) * 3
        + rows["strikes_before"].to_numpy(np.int16)
    )
    for year in sorted(int(value) for value in np.unique(row_season)):
        audit = row_season == year
        prior = profiles.loc[profiles["season"].lt(year)]
        if prior.empty:
            arrays["tm_count_profile_covered"][audit] = 0.0
            arrays["tm_count_exact_covered"][audit] = 0.0
            continue
        latest_season = prior.groupby("pitcher_id", sort=False)["season"].max()
        latest = prior.merge(
            latest_season.rename("latest_season"),
            left_on=["pitcher_id", "season"],
            right_on=["pitcher_id", "latest_season"],
            how="inner",
        )
        pitchers = row_pitcher[audit]
        counts = row_count[audit]
        exact_keys = pd.MultiIndex.from_arrays(
            [pitchers, counts], names=["pitcher_id", "count_state"]
        )
        global_keys = pd.MultiIndex.from_arrays(
            [pitchers, np.full(len(pitchers), GLOBAL_COUNT_STATE)],
            names=["pitcher_id", "count_state"],
        )
        exact = _lookup(latest, exact_keys)
        global_profile = _lookup(latest, global_keys)
        global_n = global_profile["pitch_count"].to_numpy(np.float64)
        exact_n = exact["pitch_count"].to_numpy(np.float64)
        exact_covered = np.isfinite(exact_n)
        profile_covered = np.isfinite(global_n)
        effective_n = np.where(exact_covered, exact_n, 0.0)
        reliability = effective_n / (effective_n + RELIABILITY_PRIOR_PITCHES)
        latest_values = pd.Series(pitchers).map(latest_season).to_numpy(np.float64)
        arrays["tm_count_profile_age"][audit] = year - latest_values
        arrays["tm_count_profile_covered"][audit] = profile_covered.astype(np.float32)
        arrays["tm_count_exact_covered"][audit] = exact_covered.astype(np.float32)
        arrays["tm_count_pitch_count"][audit] = effective_n
        arrays["tm_count_log_pitch_count"][audit] = np.log1p(effective_n)
        arrays["tm_count_reliability"][audit] = reliability

        for group in PITCH_GROUPS:
            column = f"share_{group}"
            exact_value = exact[column].to_numpy(np.float64)
            global_value = global_profile[column].to_numpy(np.float64)
            count_value = np.where(exact_covered, exact_value, global_value)
            arrays[f"tm_count_share_{group}"][audit] = count_value
            arrays[f"tm_global_share_{group}"][audit] = global_value
            arrays[f"tm_count_share_delta_{group}"][audit] = (
                reliability * (count_value - global_value)
            )
        for metric in METRICS:
            exact_mean = exact[f"mean_{metric}"].to_numpy(np.float64)
            global_mean = global_profile[f"mean_{metric}"].to_numpy(np.float64)
            exact_sd = exact[f"std_{metric}"].to_numpy(np.float64)
            global_sd = global_profile[f"std_{metric}"].to_numpy(np.float64)
            count_mean = np.where(exact_covered, exact_mean, global_mean)
            count_sd = np.where(exact_covered, exact_sd, global_sd)
            arrays[f"tm_count_mean_{metric}"][audit] = count_mean
            arrays[f"tm_global_mean_{metric}"][audit] = global_mean
            arrays[f"tm_count_reliable_mean_delta_{metric}"][audit] = (
                reliability * (count_mean - global_mean)
            )
            arrays[f"tm_count_sd_{metric}"][audit] = count_sd
            arrays[f"tm_global_sd_{metric}"][audit] = global_sd
            ratio = np.divide(
                count_sd,
                global_sd,
                out=np.full(len(count_sd), np.nan),
                where=np.isfinite(global_sd) & (global_sd > 0.0),
            )
            arrays[f"tm_count_reliable_sd_ratio_{metric}"][audit] = (
                reliability * (ratio - 1.0)
            )
    return pd.DataFrame(arrays, columns=names)


def fit_mask(season: np.ndarray, is_futures: np.ndarray, year: int) -> np.ndarray:
    return (season < int(year)) & ~(is_futures & (season <= 2022))


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    base_feature_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        TARGET,
        "season",
        "game_type",
        "pitcher_id",
        "balls_before",
        "strikes_before",
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    base = pd.read_parquet(base_feature_cache)
    if len(base) != len(train) or base.shape[1] != 114:
        raise ValueError("v217 base feature-cache contract mismatch")
    supplement_cache = output_dir / "count_trackman_features.parquet"
    profile_cache = output_dir / "count_trackman_profiles.parquet"
    if supplement_cache.exists() and profile_cache.exists():
        supplement = pd.read_parquet(supplement_cache)
        profiles = pd.read_parquet(profile_cache)
        cache_reused = True
    else:
        started = time.time()
        trackman = pd.read_csv(
            trackman_csv,
            usecols=[
                "pitcher_trackman_id",
                "season",
                "balls_before",
                "strikes_before",
                "pitch_type_group",
                *METRICS,
            ],
            low_memory=False,
        )
        pitcher_map = pd.read_csv(pitcher_map_csv)
        profiles = make_count_profiles(trackman, pitcher_map)
        supplement = build_count_features(train, profiles)
        profiles.to_parquet(profile_cache, index=False)
        supplement.to_parquet(supplement_cache, index=False)
        cache_reused = False
        print(
            f"[v257] profiles={len(profiles):,} supplement={supplement.shape} "
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
        checkpoint = output_dir / f"count_trackman_xgb_{year}.npy"
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
                f"[v257] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"coverage={float(supplement.loc[validation, 'tm_count_profile_covered'].mean()):.3f} "
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
                supplement.loc[validation, "tm_count_profile_covered"].mean()
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
        "cache_reused": cache_reused,
        "folds": folds,
        "eligible_for_locked_route_audit": True,
        "eligible_for_packaging": False,
        "restrictions": {
            "v217_base_hyperparameters_domain_and_weights_frozen": True,
            "strictly_prior_season_trackman_profiles": True,
            "current_row_count_only": True,
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
    parser.add_argument("--pitcher-map-csv", type=Path, required=True)
    parser.add_argument("--base-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.base_feature_cache,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
