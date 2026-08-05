"""Leakage-aware validation split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def season_holdout(df: pd.DataFrame, validation_season: int = 2024):
    """Train only on earlier seasons and validate on one future season."""
    season = df["season"].to_numpy()
    train_idx = np.flatnonzero(season < validation_season)
    valid_idx = np.flatnonzero(season == validation_season)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError(f"empty season split for validation_season={validation_season}")
    return train_idx, valid_idx


def stratified_auxiliary_split(
    df: pd.DataFrame,
    target_col: str = "control_success",
    test_size: float = 0.2,
    seed: int = 42,
):
    indices = np.arange(len(df))
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col].to_numpy(),
    )


def pitcher_group_auxiliary_split(
    df: pd.DataFrame,
    group_col: str = "pitcher_id",
    test_size: float = 0.2,
    seed: int = 42,
):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(df, groups=df[group_col]))
