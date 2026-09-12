"""Strict-forward TrackMan command-dispersion fallback XGB experiment.

This is one preregistered feature-family trial.  It keeps the v217 data
domain, weights, hyperparameters, and 114 base columns fixed.  The only
change is a train-label-free family of prior-season TrackMan repeatability
features motivated by pitching-command biomechanics: release dispersion,
within-pitch-type dispersion, within-game repeatability, pitch-mix entropy,
and within-season mechanical drift.
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


PROTOCOL = "V252_TRACKMAN_COMMAND_DISPERSION_XGB_STRICT_FORWARD_V1"
RANDOM_STATE = 2052
RAW_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
WITHIN_PITCH_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "rel_height",
    "rel_side",
)
GAME_REPEATABILITY_METRICS = (
    "rel_speed",
    "extension",
    "rel_height",
    "rel_side",
)
TREND_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "rel_height",
    "rel_side",
)


def _safe_weighted_within_sd(
    frame: pd.DataFrame, keys: list[str], metric: str
) -> pd.Series:
    """Pooled within-pitch-type SD, excluding between-type separation."""

    pitch_keys = [*keys, "pitch_type_group"]
    pieces = frame.groupby(pitch_keys, observed=True, sort=False)[metric].agg(
        ["count", "var"]
    )
    pieces["numerator"] = (pieces["count"] - 1.0) * pieces["var"].fillna(0.0)
    pooled = pieces.groupby(level=keys, sort=False)[["numerator", "count"]].sum()
    pitch_groups = pieces.groupby(level=keys, sort=False).size()
    denominator = pooled["count"] - pitch_groups
    return np.sqrt(pooled["numerator"] / denominator.where(denominator.gt(0.0)))


def make_command_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame
) -> pd.DataFrame:
    """Return one label-free command/repeatability profile per pitcher-season."""

    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(0.90), ["pitcher_id", "pitcher_trackman_id"]
    ].drop_duplicates("pitcher_trackman_id")
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    keys = ["pitcher_id", "season"]
    for column in RAW_METRICS:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined["deceleration"] = joined["rel_speed"] - joined["zone_speed"]

    grouped = joined.groupby(keys, observed=True, sort=False)
    profile = grouped.size().rename("pitch_count").to_frame()
    profile["game_count"] = grouped["trackman_game_id"].nunique()
    for metric in (*RAW_METRICS, "deceleration"):
        profile[f"sd_{metric}"] = grouped[metric].std()

    # The square-root covariance determinant is the scale of the release
    # dispersion ellipse. Correlation keeps its direction/shape information.
    release = joined.dropna(subset=["rel_height", "rel_side"]).copy()
    release["height_side"] = release["rel_height"] * release["rel_side"]
    release_group = release.groupby(keys, observed=True, sort=False)
    release_mean = release_group[["rel_height", "rel_side", "height_side"]].mean()
    covariance = (
        release_mean["height_side"]
        - release_mean["rel_height"] * release_mean["rel_side"]
    )
    var_height = release_group["rel_height"].var(ddof=0)
    var_side = release_group["rel_side"].var(ddof=0)
    determinant = (var_height * var_side - covariance.pow(2)).clip(lower=0.0)
    profile["release_ellipse_scale"] = np.sqrt(determinant)
    profile["release_height_side_corr"] = covariance / np.sqrt(
        (var_height * var_side).replace(0.0, np.nan)
    )

    for metric in WITHIN_PITCH_METRICS:
        profile[f"within_pitch_sd_{metric}"] = _safe_weighted_within_sd(
            joined.dropna(subset=[metric, "pitch_type_group"]), keys, metric
        )

    pitch_counts = joined.groupby(
        [*keys, "pitch_type_group"], observed=True, sort=False
    ).size().rename("count")
    total = pitch_counts.groupby(level=keys, sort=False).sum()
    shares = pitch_counts / total
    entropy_terms = -(shares * np.log(shares.clip(lower=1e-12)))
    profile["pitch_type_entropy"] = entropy_terms.groupby(level=keys, sort=False).sum()
    profile["pitch_type_count"] = pitch_counts.groupby(level=keys, sort=False).size()
    fastball = pitch_counts.loc[
        pitch_counts.index.get_level_values("pitch_type_group").astype(str)
        == "fastball"
    ]
    profile["fastball_share"] = (
        fastball.groupby(level=keys, sort=False).sum() / total
    )

    game_keys = [*keys, "trackman_game_id"]
    game_sd = joined.groupby(game_keys, observed=True, sort=False)[
        list(GAME_REPEATABILITY_METRICS)
    ].std()
    game_average = game_sd.groupby(level=keys, sort=False).mean()
    for metric in GAME_REPEATABILITY_METRICS:
        profile[f"within_game_sd_{metric}"] = game_average[metric]

    # OLS slope per month within a season, computed from sufficient statistics.
    month = pd.to_numeric(joined["game_month"], errors="coerce")
    joined = joined.assign(_month=month, _month_sq=month.pow(2))
    trend_group = joined.groupby(keys, observed=True, sort=False)
    month_mean = trend_group["_month"].mean()
    month_var = trend_group["_month_sq"].mean() - month_mean.pow(2)
    for metric in TREND_METRICS:
        product = joined["_month"] * joined[metric]
        product_mean = product.groupby([joined[key] for key in keys], sort=False).mean()
        product_mean.index.names = keys
        metric_mean = trend_group[metric].mean()
        covariance_month = product_mean - month_mean * metric_mean
        profile[f"monthly_slope_{metric}"] = covariance_month / month_var.where(
            month_var.gt(0.0)
        )

    profile["log_pitch_count"] = np.log1p(profile["pitch_count"])
    profile["log_game_count"] = np.log1p(profile["game_count"])
    profile = profile.drop(columns=["pitch_count", "game_count"])
    return profile.reset_index()


def profile_feature_columns(profiles: pd.DataFrame) -> list[str]:
    return [column for column in profiles.columns if column not in {"pitcher_id", "season"}]


def build_forward_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Attach latest, prior-career, and latest-minus-career profiles using season<t."""

    profile_columns = profile_feature_columns(profiles)
    output_columns = [
        name
        for column in profile_columns
        for name in (
            f"cmd_latest_{column}",
            f"cmd_career_{column}",
            f"cmd_latest_vs_career_{column}",
        )
    ] + ["cmd_profile_age", "cmd_profile_covered"]
    arrays = {
        column: np.full(len(rows), np.nan, dtype=np.float32)
        for column in output_columns
    }
    row_season = rows["season"].to_numpy(np.int16)
    row_pitcher = rows["pitcher_id"].to_numpy(np.int64)
    for year in sorted(int(value) for value in np.unique(row_season)):
        audit = row_season == year
        prior = profiles.loc[profiles["season"].lt(year)]
        if prior.empty:
            arrays["cmd_profile_covered"][audit] = 0.0
            continue
        latest = (
            prior.sort_values("season")
            .groupby("pitcher_id", sort=False)
            .tail(1)
            .set_index("pitcher_id")
        )
        career = prior.groupby("pitcher_id", sort=False)[profile_columns].mean()
        pitchers = pd.Series(row_pitcher[audit])
        latest_season = pitchers.map(latest["season"]).to_numpy(np.float64)
        arrays["cmd_profile_age"][audit] = (year - latest_season).astype(np.float32)
        arrays["cmd_profile_covered"][audit] = np.isfinite(latest_season).astype(np.float32)
        for column in profile_columns:
            latest_value = pitchers.map(latest[column]).to_numpy(np.float64)
            career_value = pitchers.map(career[column]).to_numpy(np.float64)
            arrays[f"cmd_latest_{column}"][audit] = latest_value.astype(np.float32)
            arrays[f"cmd_career_{column}"][audit] = career_value.astype(np.float32)
            arrays[f"cmd_latest_vs_career_{column}"][audit] = (
                latest_value - career_value
            ).astype(np.float32)
    return pd.DataFrame(arrays, columns=output_columns)


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
    train = pd.read_csv(
        train_csv,
        usecols=[TARGET, "season", "game_type", "pitcher_id"],
        low_memory=False,
    )
    base = pd.read_parquet(base_feature_cache)
    if len(base) != len(train) or base.shape[1] != 114:
        raise ValueError("v217 base feature-cache contract mismatch")

    supplement_cache = output_dir / "command_dispersion_features.parquet"
    profile_cache = output_dir / "command_profiles.parquet"
    if supplement_cache.exists() and profile_cache.exists():
        supplement = pd.read_parquet(supplement_cache)
        profiles = pd.read_parquet(profile_cache)
        cache_reused = True
    else:
        started = time.time()
        trackman = pd.read_csv(
            trackman_csv,
            usecols=[
                "pitcher_trackman_id", "season", "game_month",
                "trackman_game_id", "pitch_type_group", *RAW_METRICS,
            ],
            low_memory=False,
        )
        pitcher_map = pd.read_csv(pitcher_map_csv)
        profiles = make_command_profiles(trackman, pitcher_map)
        supplement = build_forward_features(train, profiles)
        profiles.to_parquet(profile_cache, index=False)
        supplement.to_parquet(supplement_cache, index=False)
        cache_reused = False
        print(
            f"[v252] profiles={len(profiles):,} supplement={supplement.shape} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    if len(supplement) != len(train):
        raise ValueError("command supplement row mismatch")
    features = pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)], axis=1
    )

    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"command_dispersion_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = fit_mask(season, is_futures, year)
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
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
                f"[v252] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"coverage={float(supplement.loc[validation, 'cmd_profile_covered'].mean()):.3f} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "coverage": float(
                supplement.loc[validation, "cmd_profile_covered"].mean()
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
        "profile_feature_count": len(profile_feature_columns(profiles)),
        "supplement_cache": str(supplement_cache),
        "cache_reused": cache_reused,
        "model_params": PARAMS,
        "random_state": RANDOM_STATE,
        "folds": folds,
        "eligible_for_locked_route_audit": True,
        "eligible_for_packaging": False,
        "restrictions": {
            "v217_base_features_frozen": True,
            "v217_xgb_hyperparameters_frozen": True,
            "v217_fit_domain_and_weights_frozen": True,
            "single_preregistered_feature_family": True,
            "strictly_prior_season_trackman_only": True,
            "no_trackman_outcome_or_location_target": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
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
