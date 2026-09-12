from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.kma_pooled_group_quantile_blend import (
    combine_year_features,
    make_pooled_design,
    make_pooled_target,
    select_plateau_mean_weight,
)


def test_make_pooled_design_stacks_groups_with_one_hot_ids() -> None:
    features = pd.DataFrame(
        {"wind": [1.0, 2.0]},
        index=pd.date_range("2023-01-01", periods=2, freq="h"),
    )
    targets = ("kpx_group_1", "kpx_group_3")
    pooled, slices = make_pooled_design(features, targets)

    assert pooled.shape == (4, 3)
    assert slices["kpx_group_1"] == slice(0, 2)
    assert slices["kpx_group_3"] == slice(2, 4)
    np.testing.assert_array_equal(
        pooled["pooled_group__kpx_group_1"],
        np.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        pooled["pooled_group__kpx_group_3"],
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    )


def test_make_pooled_target_normalizes_each_group_capacity() -> None:
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    labels = pd.DataFrame(
        {
            "kpx_group_1": [2_160.0, 10_800.0],
            "kpx_group_3": [2_100.0, 10_500.0],
        },
        index=index,
    )
    target = make_pooled_target(
        labels,
        index,
        ("kpx_group_1", "kpx_group_3"),
    )
    np.testing.assert_allclose(target, [0.1, 0.5, 0.1, 0.5])


def test_combine_year_features_preserves_time_and_schema() -> None:
    features = {
        "2022": pd.DataFrame(
            {"wind": [1.0, 2.0]},
            index=pd.date_range("2022-12-30", periods=2, freq="h"),
        ),
        "2023": pd.DataFrame(
            {"wind": [3.0, 4.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        ),
    }

    combined = combine_year_features(features, ("2022", "2023"))

    assert len(combined) == 4
    assert combined.index.is_monotonic_increasing
    assert combined["wind"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_combine_year_features_rejects_schema_drift() -> None:
    index = pd.date_range("2022-12-30", periods=2, freq="h")
    features = {
        "2022": pd.DataFrame({"wind": [1.0, 2.0]}, index=index),
        "2023": pd.DataFrame({"speed": [1.0, 2.0]}, index=index),
    }

    with np.testing.assert_raises_regex(
        ValueError, "training feature columns differ"
    ):
        combine_year_features(features, ("2022", "2023"))


def test_select_plateau_mean_weight_averages_q1_near_best_grid() -> None:
    records = [
        {
            "weight": weight,
            "delta": {
                "score": score,
                "one_minus_nmae": 0.001,
                "ficr": 0.001,
            },
            "eligible": True,
        }
        for weight, score in (
            (0.0375, 0.00191),
            (0.040625, 0.00193),
            (0.04375, 0.00198),
            (0.046875, 0.00160),
        )
    ]
    selected, details = select_plateau_mean_weight(records)
    np.testing.assert_allclose(selected["weight"], 0.040625)
    assert details["near_best_weights"] == [0.0375, 0.040625, 0.04375]
    np.testing.assert_allclose(details["plateau_mean_weight"], 0.040625)
