"""Standalone row-local runtime for the five-seed recent-F expert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


TARGET = "control_success"
CAT_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]
SEEDS = (2871, 2873, 2875, 2877, 2879)


def build_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.drop(columns=["row_id", TARGET, "season"], errors="ignore").copy()
    for column in CAT_COLUMNS:
        output[column] = output[column].fillna("__NA__").astype(str)
    for column in output.columns.difference(CAT_COLUMNS):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(-999.0)
    output["count_code"] = (
        frame["balls_before"].astype(str)
        + "-"
        + frame["strikes_before"].astype(str)
    )
    output["same_hand"] = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).astype(str)
    output["pressure_code"] = (
        frame["num_runners_on"].gt(0) | frame["li"].ge(1.5)
    ).astype(str)
    return output.reindex(columns=columns)


def predict(frame: pd.DataFrame, asset: Path) -> np.ndarray:
    columns = json.loads((asset / "feature_columns.json").read_text(encoding="utf-8"))
    selected = frame["game_type"].astype(str).eq("F").to_numpy()
    output = np.full(len(frame), np.nan, dtype=np.float64)
    if not np.any(selected):
        return output
    features = build_features(frame.loc[selected].reset_index(drop=True), columns)
    predictions = []
    for seed in SEEDS:
        model = CatBoostClassifier()
        model.load_model(asset / f"futures_expert_seed{seed}.cbm")
        predictions.append(model.predict_proba(features)[:, 1])
    output[selected] = np.column_stack(predictions).mean(axis=1)
    return output
