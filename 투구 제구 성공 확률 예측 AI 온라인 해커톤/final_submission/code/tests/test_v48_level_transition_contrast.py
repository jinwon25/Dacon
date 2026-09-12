import numpy as np
import pandas as pd

from src.archive.v48_level_transition_contrast import (
    previous_level,
    residual_contrast,
    transition_key,
)


def _train():
    return pd.DataFrame(
        {
            "season": [2021] * 6 + [2022] * 4,
            "game_type": ["R", "R", "R", "F", "F", "F", "F", "R", "F", "R"],
            "pitcher_id": [1, 1, 1, 2, 2, 2, 1, 2, 3, 3],
            "domain3": ["R_CORE"] * 3 + ["F"] * 4 + ["R_CORE", "F", "R_CORE"],
        }
    )


def test_previous_level_and_transition_use_prior_season_only():
    train = _train()
    levels = previous_level(train, 2022, "pitcher_id")
    assert levels.loc[1] == "R"
    assert levels.loc[2] == "F"
    rows = train.loc[train["season"].eq(2022)].reset_index(drop=True)
    assert transition_key(train, rows, 2022, "pitcher_id").tolist() == [
        "R>F",
        "F>R",
        "NEW>F",
        "NEW>R",
    ]


def test_residual_contrast_is_domain_centered_and_query_label_free():
    train = _train()
    source = train.loc[train["season"].eq(2022)].reset_index(drop=True)
    query = source.copy()
    target = np.array([1.0, 0.0, 1.0, 0.0])
    parent = np.full(4, 0.5)
    first = residual_contrast(
        train,
        source,
        target,
        parent,
        query,
        source_year=2022,
        query_year=2022,
        entity="pitcher_id",
        alpha=10.0,
    )
    changed = query.copy()
    changed["unused_target"] = [0, 1, 0, 1]
    second = residual_contrast(
        train,
        source,
        target,
        parent,
        changed,
        source_year=2022,
        query_year=2022,
        entity="pitcher_id",
        alpha=10.0,
    )
    assert np.allclose(first, second)
    assert np.isfinite(first).all()
