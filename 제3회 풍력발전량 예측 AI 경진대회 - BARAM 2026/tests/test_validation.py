import numpy as np
import pandas as pd

from src.validation import assign_oof_roles, make_expanding_year_folds


def test_expanding_year_folds_are_strictly_past_to_future() -> None:
    index = pd.date_range("2022-01-01 01:00:00", "2024-12-31 01:00:00", freq="D")
    labels = pd.Series(1.0, index=index)

    folds = make_expanding_year_folds(index, labels, "kpx_group_1")

    assert [fold.valid_year for fold in folds] == [2023, 2024]
    assert folds[0].train_years == (2022,)
    assert folds[1].train_years == (2022, 2023)
    for fold in folds:
        fold.assert_causal(index)


def test_group3_like_missing_first_year_yields_one_fold() -> None:
    index = pd.date_range("2022-01-01 01:00:00", "2024-12-31 01:00:00", freq="D")
    labels = pd.Series(np.where(index.year == 2022, np.nan, 1.0), index=index)

    folds = make_expanding_year_folds(index, labels, "kpx_group_3")

    assert len(folds) == 1
    assert folds[0].train_years == (2023,)
    assert folds[0].valid_year == 2024


def test_oof_roles_keep_evaluation_separate() -> None:
    index = pd.to_datetime(["2023-03-01 01:00:00", "2023-09-01 01:00:00", "2024-03-01 01:00:00", "2025-01-01 00:00:00"])
    assert assign_oof_roles(index).tolist() == ["selection", "calibration", "evaluation", "evaluation"]


def test_next_year_midnight_stays_in_prior_forecast_year_fold() -> None:
    index = pd.to_datetime(["2022-01-01 01:00:00", "2023-01-01 00:00:00", "2023-01-01 01:00:00", "2024-01-01 00:00:00"])
    labels = pd.Series(1.0, index=index)

    folds = make_expanding_year_folds(index, labels, "kpx_group_1")

    assert len(folds) == 1
    assert folds[0].train.sum() == 2
    assert folds[0].valid.sum() == 2
