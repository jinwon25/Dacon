from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.nested_quantile_base import (
    Selection,
    choose_selection,
    curtailment_mask,
    make_nested_folds,
)


def _season_rows(start: str, periods: int = 90 * 24) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h")


def test_nested_folds_keep_outer_season_out_of_inner_selection() -> None:
    index = _season_rows("2022-01-01", periods=3 * 365 * 24)
    issue = pd.DatetimeIndex(index.floor("D") + pd.Timedelta(hours=13))
    available = np.ones(len(index), dtype=bool)
    folds = make_nested_folds(
        index,
        issue,
        available,
        minimum_train_rows=1_000,
        minimum_valid_rows=100,
    )
    assert [fold.name for fold in folds] == [
        "2024-DJF",
        "2024-MAM",
        "2024-JJA",
        "2024-SON",
    ]
    for fold in folds:
        assert not np.any(fold.train & fold.valid)
        assert not np.any(fold.inner_train & fold.inner_valid)
        assert np.all(fold.inner_train <= fold.train)
        assert np.all(fold.inner_valid <= fold.train)
        for issue_value in np.unique(issue):
            in_train = np.any(fold.train & (issue == issue_value))
            in_valid = np.any(fold.valid & (issue == issue_value))
            assert not (in_train and in_valid)


def test_selection_prefers_score_then_ficr() -> None:
    lower_ficr = Selection("quantile", 0.60, "all", 300, 0.61, 0.86, 0.36)
    higher_ficr = Selection("quantile", 0.65, "all", 320, 0.61, 0.85, 0.37)
    assert choose_selection([lower_ficr, higher_ficr]) == higher_ficr


def test_curtailment_mask_only_flags_training_rows() -> None:
    wind = np.repeat(np.arange(5.0, 10.0, 0.5), 30)
    y = np.clip((wind - 4.0) / 8.0, 0.0, 1.0) * 21_000.0
    training = np.ones(len(y), dtype=bool)
    curtailed_row = 100
    validation_row = 101
    y[curtailed_row] = 100.0
    y[validation_row] = 100.0
    training[validation_row] = False
    mask = curtailment_mask(y, wind, training, 21_000.0)
    assert mask[curtailed_row]
    assert not mask[validation_row]
    assert not np.any(mask & ~training)
