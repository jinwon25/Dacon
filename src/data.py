"""Memory-conscious CSV loading and schema constants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

MAIN_STRING_COLS = ["row_id"]
MAIN_CATEGORY_COLS = ["top_bottom", "game_type", "base_state"]
MAIN_FLOAT_COLS = [
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

TRACKMAN_CATEGORY_COLS = [
    "game_date",
    "trackman_game_id",
    "top_bottom",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
    "tagged_pitch_type",
    "auto_pitch_type",
    "pitch_type_group",
]
TRACKMAN_FLOAT_COLS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]


def csv_columns(path: str | Path) -> list[str]:
    return pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()


def main_dtypes(path: str | Path) -> dict[str, str]:
    columns = csv_columns(path)
    dtypes: dict[str, str] = {}
    for col in columns:
        if col in MAIN_STRING_COLS:
            dtypes[col] = "string"
        elif col in MAIN_CATEGORY_COLS:
            dtypes[col] = "category"
        elif col in MAIN_FLOAT_COLS:
            dtypes[col] = "float32"
        else:
            dtypes[col] = "int32"
    return dtypes


def trackman_dtypes(path: str | Path) -> dict[str, str]:
    columns = csv_columns(path)
    dtypes: dict[str, str] = {}
    for col in columns:
        if col in TRACKMAN_CATEGORY_COLS:
            dtypes[col] = "category"
        elif col in TRACKMAN_FLOAT_COLS:
            dtypes[col] = "float32"
        else:
            dtypes[col] = "int32"
    return dtypes


def read_main(
    path: str | Path,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    dtypes = main_dtypes(path)
    if usecols is not None:
        dtypes = {col: dtype for col, dtype in dtypes.items() if col in usecols}
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=dtypes,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def read_trackman(
    path: str | Path,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    dtypes = trackman_dtypes(path)
    if usecols is not None:
        dtypes = {col: dtype for col, dtype in dtypes.items() if col in usecols}
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=dtypes,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def iter_csv(path: str | Path, kind: str, chunksize: int = 200_000):
    dtype = main_dtypes(path) if kind == "main" else trackman_dtypes(path)
    yield from pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    )
