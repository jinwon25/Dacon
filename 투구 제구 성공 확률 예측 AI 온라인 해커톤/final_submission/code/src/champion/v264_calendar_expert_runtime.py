"""Standalone row-local runtime for v261 command and batter experts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


def _command_profile_columns(profiles: pd.DataFrame) -> list[str]:
    return [
        column for column in profiles.columns if column not in {"pitcher_id", "season"}
    ]


def build_command_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    profile_columns = _command_profile_columns(profiles)
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
        arrays["cmd_profile_age"][audit] = year - latest_season
        arrays["cmd_profile_covered"][audit] = np.isfinite(latest_season).astype(
            np.float32
        )
        for column in profile_columns:
            latest_value = pitchers.map(latest[column]).to_numpy(np.float64)
            career_value = pitchers.map(career[column]).to_numpy(np.float64)
            arrays[f"cmd_latest_{column}"][audit] = latest_value
            arrays[f"cmd_career_{column}"][audit] = career_value
            arrays[f"cmd_latest_vs_career_{column}"][audit] = (
                latest_value - career_value
            )
    return pd.DataFrame(arrays, columns=output_columns)


def _batter_profile_columns(profiles: pd.DataFrame) -> list[str]:
    return [
        column for column in profiles.columns if column not in {"batter_id", "season"}
    ]


def build_batter_features(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    profile_columns = _batter_profile_columns(profiles)
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


def _predict_one(
    base: pd.DataFrame,
    supplement: pd.DataFrame,
    asset: Path,
    model_name: str,
    columns_name: str,
) -> np.ndarray:
    columns = json.loads((asset / columns_name).read_text(encoding="utf-8"))
    features = pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)],
        axis=1,
        copy=False,
    ).reindex(columns=columns)
    model = xgb.XGBClassifier()
    model.load_model(asset / model_name)
    return model.predict_proba(features)[:, 1].astype(np.float64)


def predict(
    frame: pd.DataFrame,
    asset: Path,
    base_runtime: object,
    base_asset: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return command and batter probabilities using only current-row fields."""

    base = base_runtime.build(frame, base_asset)
    command_profiles = joblib.load(asset / "command_profiles.joblib")
    batter_profiles = joblib.load(asset / "batter_profiles.joblib")
    command = build_command_features(frame, command_profiles)
    batter = build_batter_features(frame, batter_profiles)
    command_probability = _predict_one(
        base,
        command,
        asset,
        "command_expert_xgb.json",
        "command_feature_columns.json",
    )
    batter_probability = _predict_one(
        base,
        batter,
        asset,
        "batter_expert_xgb.json",
        "batter_feature_columns.json",
    )
    return command_probability, batter_probability
