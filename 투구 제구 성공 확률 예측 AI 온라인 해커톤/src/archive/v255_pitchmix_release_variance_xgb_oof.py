"""Strict-forward pitch-mix-weighted TrackMan release-variance XGB trial."""

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


PROTOCOL = "V255_PITCHMIX_RELEASE_VARIANCE_XGB_STRICT_FORWARD_V1"
RANDOM_STATE = 2055
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
MIX_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
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


def make_pitch_type_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame
) -> pd.DataFrame:
    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(0.90), ["pitcher_id", "pitcher_trackman_id"]
    ].drop_duplicates("pitcher_trackman_id")
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    joined = joined.loc[joined["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    for metric in METRICS:
        joined[metric] = pd.to_numeric(joined[metric], errors="coerce")
    grouped = joined.groupby(
        ["pitcher_id", "season", "pitch_type_group"], observed=True, sort=False
    )
    means = grouped[list(METRICS)].mean().add_prefix("mean_")
    stds = grouped[list(METRICS)].std().add_prefix("sd_")
    count = grouped.size().rename("pitch_count")
    return pd.concat([means, stds, count], axis=1).reset_index()


def _latest_wide(prior: pd.DataFrame) -> pd.DataFrame:
    latest_season = (
        prior.groupby("pitcher_id", sort=False)["season"].max().rename("latest_season")
    )
    latest = prior.merge(
        latest_season, left_on=["pitcher_id", "season"],
        right_on=["pitcher_id", "latest_season"], how="inner",
    )
    value_columns = [column for column in latest if column.startswith(("mean_", "sd_"))]
    value_columns.append("pitch_count")
    wide = latest.pivot_table(
        index="pitcher_id", columns="pitch_type_group", values=value_columns,
        aggfunc="first",
    )
    wide.columns = [f"{value}__{group}" for value, group in wide.columns]
    return latest_season.to_frame().join(wide, how="left")


def build_pitchmix_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Use current row's ASOF mix with pitcher profiles strictly before its season."""

    feature_names = []
    for metric in METRICS:
        feature_names.extend([
            f"mix_expected_mean_{metric}",
            f"mix_within_sd_{metric}",
            f"mix_total_sd_{metric}",
            f"mix_between_sd_{metric}",
            f"mix_type_range_{metric}",
            f"mix_fb_vs_nonfb_{metric}",
        ])
        for group in PITCH_GROUPS:
            feature_names.extend([
                f"mix_{group}_mean_{metric}", f"mix_{group}_sd_{metric}"
            ])
    feature_names.extend([
        "mix_profile_age", "mix_profile_covered", "mix_group_coverage",
        "mix_weight_coverage", "mix_prior_fastball_share",
        "mix_prior_breaking_share", "mix_prior_offspeed_share",
    ])
    arrays = {
        name: np.full(len(rows), np.nan, dtype=np.float32) for name in feature_names
    }
    row_season = rows["season"].to_numpy(np.int16)
    row_pitcher = rows["pitcher_id"].to_numpy(np.int64)
    for year in sorted(int(value) for value in np.unique(row_season)):
        audit = row_season == year
        prior = profiles.loc[profiles["season"].lt(year)]
        if prior.empty:
            arrays["mix_profile_covered"][audit] = 0.0
            continue
        wide = _latest_wide(prior)
        pitchers = pd.Series(row_pitcher[audit])
        latest_season = pitchers.map(wide["latest_season"]).to_numpy(np.float64)
        covered = np.isfinite(latest_season)
        arrays["mix_profile_age"][audit] = (year - latest_season).astype(np.float32)
        arrays["mix_profile_covered"][audit] = covered.astype(np.float32)

        counts = np.column_stack([
            pitchers.map(wide.get(f"pitch_count__{group}", pd.Series(dtype=float)))
            .fillna(0.0).to_numpy(np.float64)
            for group in PITCH_GROUPS
        ])
        prior_total = counts.sum(axis=1, keepdims=True)
        prior_share = np.divide(
            counts, prior_total, out=np.zeros_like(counts), where=prior_total > 0.0
        )
        for index, group in enumerate(PITCH_GROUPS):
            arrays[f"mix_prior_{group}_share"][audit] = prior_share[:, index]

        current = rows.loc[audit, list(MIX_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float64)
        current_sum = np.nansum(current, axis=1, keepdims=True)
        current_valid = np.isfinite(current).all(axis=1) & (current_sum[:, 0] > 0.0)
        weights = prior_share.copy()
        weights[current_valid] = current[current_valid] / current_sum[current_valid]
        group_available = counts > 0.0
        effective = weights * group_available
        effective_sum = effective.sum(axis=1, keepdims=True)
        effective = np.divide(
            effective, effective_sum, out=np.zeros_like(effective),
            where=effective_sum > 0.0,
        )
        arrays["mix_group_coverage"][audit] = group_available.sum(axis=1)
        arrays["mix_weight_coverage"][audit] = (
            weights * group_available
        ).sum(axis=1)

        for metric in METRICS:
            means = np.column_stack([
                pitchers.map(wide.get(f"mean_{metric}__{group}", pd.Series(dtype=float)))
                .to_numpy(np.float64)
                for group in PITCH_GROUPS
            ])
            sds = np.column_stack([
                pitchers.map(wide.get(f"sd_{metric}__{group}", pd.Series(dtype=float)))
                .to_numpy(np.float64)
                for group in PITCH_GROUPS
            ])
            valid = np.isfinite(means) & np.isfinite(sds)
            metric_weights = effective * valid
            weight_sum = metric_weights.sum(axis=1, keepdims=True)
            metric_weights = np.divide(
                metric_weights, weight_sum, out=np.zeros_like(metric_weights),
                where=weight_sum > 0.0,
            )
            safe_means = np.nan_to_num(means)
            safe_sds = np.nan_to_num(sds)
            expected = (metric_weights * safe_means).sum(axis=1)
            within_var = (metric_weights * np.square(safe_sds)).sum(axis=1)
            between_var = (
                metric_weights * np.square(safe_means - expected[:, None])
            ).sum(axis=1)
            missing = weight_sum[:, 0] <= 0.0
            expected[missing] = np.nan
            within_var[missing] = np.nan
            between_var[missing] = np.nan
            arrays[f"mix_expected_mean_{metric}"][audit] = expected
            arrays[f"mix_within_sd_{metric}"][audit] = np.sqrt(within_var)
            arrays[f"mix_between_sd_{metric}"][audit] = np.sqrt(between_var)
            arrays[f"mix_total_sd_{metric}"][audit] = np.sqrt(within_var + between_var)
            with np.errstate(all="ignore"):
                type_range = np.nanmax(means, axis=1) - np.nanmin(means, axis=1)
            type_range[np.sum(np.isfinite(means), axis=1) < 2] = np.nan
            arrays[f"mix_type_range_{metric}"][audit] = type_range
            nonfb_weight = metric_weights[:, 1:].sum(axis=1)
            nonfb_mean = np.divide(
                (metric_weights[:, 1:] * safe_means[:, 1:]).sum(axis=1),
                nonfb_weight, out=np.full(len(nonfb_weight), np.nan),
                where=nonfb_weight > 0.0,
            )
            fb_gap = means[:, 0] - nonfb_mean
            arrays[f"mix_fb_vs_nonfb_{metric}"][audit] = fb_gap
            for index, group in enumerate(PITCH_GROUPS):
                arrays[f"mix_{group}_mean_{metric}"][audit] = means[:, index]
                arrays[f"mix_{group}_sd_{metric}"][audit] = sds[:, index]
    return pd.DataFrame(arrays, columns=feature_names)


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
    usecols = [TARGET, "season", "game_type", "pitcher_id", *MIX_COLUMNS]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    base = pd.read_parquet(base_feature_cache)
    if len(base) != len(train) or base.shape[1] != 114:
        raise ValueError("v217 base feature-cache contract mismatch")
    supplement_cache = output_dir / "pitchmix_release_features.parquet"
    profile_cache = output_dir / "pitch_type_profiles.parquet"
    if supplement_cache.exists() and profile_cache.exists():
        supplement = pd.read_parquet(supplement_cache)
        profiles = pd.read_parquet(profile_cache)
        cache_reused = True
    else:
        started = time.time()
        trackman = pd.read_csv(
            trackman_csv,
            usecols=["pitcher_trackman_id", "season", "pitch_type_group", *METRICS],
            low_memory=False,
        )
        pitcher_map = pd.read_csv(pitcher_map_csv)
        profiles = make_pitch_type_profiles(trackman, pitcher_map)
        supplement = build_pitchmix_features(train, profiles)
        profiles.to_parquet(profile_cache, index=False)
        supplement.to_parquet(supplement_cache, index=False)
        cache_reused = False
        print(
            f"[v255] profiles={len(profiles):,} supplement={supplement.shape} "
            f"elapsed={time.time()-started:.1f}s", flush=True,
        )
    features = pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)], axis=1
    )
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"pitchmix_release_xgb_{year}.npy"
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
                f"[v255] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"coverage={float(supplement.loc[validation, 'mix_profile_covered'].mean()):.3f} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s", flush=True,
            )
        if len(prediction) != int(validation.sum()):
            raise ValueError(f"checkpoint length mismatch: {year}")
        folds[str(year)] = {
            "fit_rows": int(fit.sum()), "audit_rows": int(validation.sum()),
            "coverage": float(supplement.loc[validation, "mix_profile_covered"].mean()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed), "reused": reused,
            "checkpoint": str(checkpoint),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "strict_forward_oof_fitted",
        "family_trial_count": 1,
        "base_feature_count": 114,
        "added_feature_count": int(supplement.shape[1]),
        "feature_count": int(features.shape[1]),
        "pitch_type_profile_rows": int(len(profiles)),
        "cache_reused": cache_reused,
        "folds": folds,
        "eligible_for_locked_route_audit": True,
        "eligible_for_packaging": False,
        "restrictions": {
            "v217_base_hyperparameters_domain_and_weights_frozen": True,
            "strictly_prior_season_trackman_profiles": True,
            "current_row_asof_pitchmix_only": True,
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
        args.train_csv, args.trackman_csv, args.pitcher_map_csv,
        args.base_feature_cache, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
