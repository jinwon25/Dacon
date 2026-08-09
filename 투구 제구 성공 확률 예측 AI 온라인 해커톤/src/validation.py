"""Leakage-aware validation split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class OuterFold:
    validation_season: int
    train_idx: np.ndarray
    valid_idx: np.ndarray

    @property
    def name(self) -> str:
        return f"train_before_{self.validation_season}_validate_{self.validation_season}"


def season_holdout(df: pd.DataFrame, validation_season: int = 2024):
    """Train only on earlier seasons and validate on one future season."""
    season = df["season"].to_numpy()
    train_idx = np.flatnonzero(season < validation_season)
    valid_idx = np.flatnonzero(season == validation_season)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError(f"empty season split for validation_season={validation_season}")
    return train_idx, valid_idx


def walk_forward_splits(
    df: pd.DataFrame,
    validation_seasons: tuple[int, ...] = (2021, 2022, 2023, 2024),
) -> list[OuterFold]:
    """Return expanding-window season folds with strict past-only training."""
    season = pd.to_numeric(df["season"], errors="raise").to_numpy(dtype=np.int32)
    folds: list[OuterFold] = []
    for validation_season in validation_seasons:
        train_idx = np.flatnonzero(season < validation_season)
        valid_idx = np.flatnonzero(season == validation_season)
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(f"empty walk-forward fold for {validation_season}")
        if int(season[train_idx].max()) >= validation_season:
            raise AssertionError("outer fold contains non-past training rows")
        if not np.all(season[valid_idx] == validation_season):
            raise AssertionError("outer validation season mismatch")
        folds.append(OuterFold(validation_season, train_idx, valid_idx))
    return folds


def historical_oof_indices(
    df: pd.DataFrame,
    outer_validation_season: int,
    minimum_train_seasons: int = 1,
) -> list[OuterFold]:
    """Inner forward folds available strictly before one outer validation year."""
    seasons = sorted(
        int(value)
        for value in pd.unique(df.loc[df["season"] < outer_validation_season, "season"])
    )
    validation_seasons = tuple(seasons[minimum_train_seasons:])
    if not validation_seasons:
        return []
    return walk_forward_splits(df, validation_seasons)


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
