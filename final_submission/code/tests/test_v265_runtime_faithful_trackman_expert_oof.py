import numpy as np
import pandas as pd
import pytest

from src.archive.v265_runtime_faithful_trackman_expert_oof import (
    combine_features,
    fold_masks,
)


def test_combine_features_preserves_order_and_rows() -> None:
    base = pd.DataFrame({"base_a": [1.0, 2.0], "base_b": [3.0, 4.0]}, index=[5, 7])
    supplement = pd.DataFrame({"extra": [8.0, 9.0]}, index=[5, 7])
    combined = combine_features(base, supplement)
    assert list(combined.columns) == ["base_a", "base_b", "extra"]
    assert combined.index.tolist() == [0, 1]
    np.testing.assert_allclose(combined["extra"], [8.0, 9.0])


def test_combine_features_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        combine_features(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}))


def test_fold_masks_keep_strict_history_and_exclude_old_futures() -> None:
    season = np.array([2021, 2022, 2022, 2023, 2023])
    futures = np.array([False, False, True, False, True])
    fit, validation = fold_masks(season, futures, 2023)
    np.testing.assert_array_equal(fit, [True, True, False, False, False])
    np.testing.assert_array_equal(validation, [False, False, False, True, False])
