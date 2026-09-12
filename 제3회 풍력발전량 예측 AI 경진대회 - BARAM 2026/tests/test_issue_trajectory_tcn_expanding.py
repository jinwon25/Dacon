from __future__ import annotations

import numpy as np

from experiments.issue_trajectory_tcn import SequenceBundle
from experiments.issue_trajectory_tcn_expanding import (
    expanding_fold_rows,
    subset_bundle,
)


def test_expanding_folds_train_only_before_query() -> None:
    starts = np.arange(
        np.datetime64("2023-01-01"),
        np.datetime64("2025-01-01"),
        np.timedelta64(1, "D"),
    )
    timestamps = np.stack(
        [start + np.arange(24).astype("timedelta64[h]") for start in starts]
    )
    folds = expanding_fold_rows(timestamps)
    assert [fold["name"] for fold in folds] == ["q1", "q2", "q3", "q4"]
    for fold in folds:
        train_end = timestamps[fold["train"], 0].max()
        query_start = timestamps[fold["query"], 0].min()
        assert train_end < query_start


def test_subset_bundle_preserves_aligned_arrays() -> None:
    bundle = SequenceBundle(
        features=np.zeros((3, 24, 2), dtype=np.float32),
        targets=np.zeros((3, 24, 3), dtype=np.float32),
        target_mask=np.ones((3, 24, 3), dtype=bool),
        timestamps=np.zeros((3, 24), dtype="datetime64[ns]"),
        issues=np.zeros(3, dtype="datetime64[ns]"),
    )
    selected = subset_bundle(bundle, np.array([True, False, True]))
    assert selected.features.shape == (2, 24, 2)
    assert selected.targets is not None
    assert selected.targets.shape == (2, 24, 3)
