"""Standalone row-local inference helper for the v197 hierarchy model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


CATEGORICAL_BASE = (
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
    "game_dayofweek",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load H1 runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hierarchical_features_for_inference(
    frame: pd.DataFrame,
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Recreate v184 hierarchy from one row and a frozen opening snapshot."""
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    career_n = (
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
        .fillna(0.0).to_numpy(np.float64)
    )
    career_rate = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(0.5).to_numpy(np.float64)
    )
    career_events = career_n * career_rate
    opening = snapshot["pitcher_opening"]
    previous_n = np.fromiter(
        (float(opening.get(int(key), (0.0, 0.0))[0]) for key in pitcher),
        dtype=np.float64, count=len(frame),
    )
    previous_events = np.fromiter(
        (float(opening.get(int(key), (0.0, 0.0))[1]) for key in pitcher),
        dtype=np.float64, count=len(frame),
    )
    league_prior = np.full(len(frame), float(snapshot["league_prior"]), dtype=np.float64)
    season_n = np.maximum(career_n - previous_n, 0.0)
    season_events = np.clip(career_events - previous_events, 0.0, season_n)
    raw_season_rate = np.divide(
        season_events, season_n, out=league_prior.copy(), where=season_n > 0.0
    )
    recent_columns = (
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    )
    recent = np.column_stack(
        [pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
         for column in recent_columns]
    )
    fallback = np.broadcast_to(career_rate[:, None], recent.shape)
    recent = np.where(np.isfinite(recent), recent, fallback)
    recent_mean = np.mean(recent, axis=1)
    recent_std = np.std(recent, axis=1)
    career_strength = np.clip(
        55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(career_n)),
        50.0, 180.0,
    )
    career_base = (
        career_events + career_strength * league_prior
    ) / np.maximum(career_n + career_strength, 1.0)
    season_strength = np.clip(30.0 + 160.0 * recent_std, 25.0, 100.0)
    season_base = (
        season_events + season_strength * career_base
    ) / np.maximum(season_n + season_strength, 1.0)
    season_reliability = season_n / (season_n + 80.0)
    season_weight = 0.15 + 0.30 * season_reliability
    hierarchical = career_base + season_weight * (season_base - career_base)
    return pd.DataFrame(
        {
            "hier_league_prior": league_prior,
            "hier_career_base": career_base,
            "hier_season_raw": raw_season_rate,
            "hier_season_base": season_base,
            "hier_prediction": hierarchical,
            "hier_previous_n_log": np.log1p(previous_n),
            "hier_season_n_log": np.log1p(season_n),
            "hier_season_reliability": season_reliability,
            "hier_career_strength": career_strength,
            "hier_season_strength": season_strength,
            "hier_recent_mean": recent_mean,
            "hier_recent_std": recent_std,
            "hier_recent_minus_career": recent_mean - career_rate,
            "hier_season_minus_career": season_base - career_base,
        },
        index=frame.index,
        dtype=np.float32,
    )


def prepare_model_frame(
    frame: pd.DataFrame,
    h1_features: list[str],
    hierarchical: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_names = list(dict.fromkeys(h1_features))
    output = frame.loc[:, feature_names].copy()
    for column in hierarchical.columns:
        output[column] = hierarchical[column].to_numpy(np.float32)
        feature_names.append(column)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="coerce").fillna(-1).astype(int)
    batter = pd.to_numeric(frame["batter_id"], errors="coerce").fillna(-1).astype(int)
    phand = pd.to_numeric(frame["pitcher_hand"], errors="coerce").fillna(-1).astype(int)
    bhand = pd.to_numeric(frame["batter_hand"], errors="coerce").fillna(-1).astype(int)
    base = frame["base_state"].fillna("NA").astype(str)
    interactions = {
        "cat_pitcher_count": pitcher.astype(str) + "|" + balls.astype(str) + "-" + strikes.astype(str),
        "cat_pitcher_platoon": pitcher.astype(str) + "|" + bhand.astype(str),
        "cat_batter_platoon": batter.astype(str) + "|" + phand.astype(str),
        "cat_count_base": balls.astype(str) + "-" + strikes.astype(str) + "|" + base,
        "cat_hand_count": phand.astype(str) + "-" + bhand.astype(str) + "|" + balls.astype(str) + "-" + strikes.astype(str),
    }
    for column, values in interactions.items():
        output[column] = values
        feature_names.append(column)
    categorical = [column for column in CATEGORICAL_BASE if column in feature_names]
    categorical.extend(interactions)
    categorical = list(dict.fromkeys(categorical))
    for column in categorical:
        output[column] = output[column].fillna("NA").astype(str)
    numeric = [column for column in feature_names if column not in categorical]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return output, feature_names, categorical


def combine_hierarchical_residual(
    hierarchical_base: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(hierarchical_base, dtype=np.float64)
        + np.asarray(residual, dtype=np.float64),
        0.001,
        0.999,
    )


def predict(frame: pd.DataFrame, h1_root: Path, model_root: Path) -> np.ndarray:
    snapshot = joblib.load(model_root / "snapshot.joblib")
    h1_module = _load_module("v197_h1_feature_runtime", h1_root / "script.py")
    bundle = joblib.load(h1_root / "model" / "rf.pkl")
    prepared = h1_module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in h1_module.CAAFE_COLS):
        prepared = h1_module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in h1_module.ASOF_COLS):
        prepared = h1_module.attach_asof_state(prepared, bundle)
    h1_frame = h1_module.build_features(prepared, bundle)
    if list(h1_frame.columns) != list(snapshot["h1_features"]):
        raise ValueError("v197 H1 feature contract mismatch")
    hierarchical = hierarchical_features_for_inference(h1_frame, snapshot)
    model_frame, feature_names, categorical = prepare_model_frame(
        h1_frame, list(snapshot["h1_features"]), hierarchical
    )
    if feature_names != list(snapshot["feature_names"]):
        raise ValueError("v197 model feature order mismatch")
    if categorical != list(snapshot["categorical"]):
        raise ValueError("v197 categorical feature order mismatch")
    model = CatBoostRegressor()
    model.load_model(model_root / "hier_context.cbm")
    residual = model.predict(
        Pool(model_frame[feature_names], cat_features=categorical)
    )
    return combine_hierarchical_residual(
        hierarchical["hier_prediction"].to_numpy(np.float64), residual
    )
