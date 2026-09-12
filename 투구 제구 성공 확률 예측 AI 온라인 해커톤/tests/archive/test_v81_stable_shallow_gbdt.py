import numpy as np
import pandas as pd

from src.archive.v81_stable_shallow_gbdt import (
    TreeSpec,
    predict_correction,
    rolling_month_splits,
)


def test_rolling_month_splits_are_forward_and_nonoverlapping() -> None:
    frame = pd.DataFrame({"game_month": np.repeat(np.arange(3, 11), 1200)})
    config = {
        "minimum_fold_rows": 1000,
        "rolling_source_folds": [
            {"train_month_max": 5, "valid_month_min": 6, "valid_month_max": 7},
            {"train_month_max": 7, "valid_month_min": 8, "valid_month_max": 12},
            {"train_month_max": 8, "valid_month_min": 9, "valid_month_max": 9},
            {"train_month_max": 9, "valid_month_min": 10, "valid_month_max": 12},
        ],
    }
    splits = rolling_month_splits(frame, config)
    covered = np.zeros(len(frame), dtype=bool)
    for train, valid in splits:
        assert frame.loc[train, "game_month"].max() < frame.loc[valid, "game_month"].min()
        assert not np.any(covered & valid)
        covered |= valid


def test_empty_tree_spec_predicts_zero() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0]})
    spec = TreeSpec(feature_names=(), model=None, risk="uniform")
    assert predict_correction(spec, features, 0.03).tolist() == [0.0, 0.0]
