import numpy as np
import pandas as pd

from experiments.cross_group_trajectory_smoothing import (
    driver_consensus_mask,
    shift_within_issue_cycle,
    smooth_group_3,
    triangular_smooth,
)


def test_shift_does_not_cross_issue_cycle_boundary() -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01 23:00", "2025-01-02 00:00", "2025-01-02 01:00"]
    )
    values = np.asarray([1.0, 2.0, 100.0])

    shifted = shift_within_issue_cycle(values, timestamps, 1)

    np.testing.assert_allclose(shifted, [2.0, 2.0, 100.0])


def test_triangular_smooth_uses_current_value_at_missing_boundaries() -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"]
    )
    values = np.asarray([0.0, 4.0, 0.0])

    result = triangular_smooth(values, timestamps)

    np.testing.assert_allclose(result, [1.0, 2.0, 1.0])


def test_driver_consensus_requires_matching_delta_signs() -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"]
    )
    group_1 = np.asarray([0.0, 10_000.0, 0.0])
    group_2_same = np.asarray([0.0, 8_000.0, 0.0])
    group_2_opposite = np.asarray([8_000.0, 0.0, 8_000.0])

    assert driver_consensus_mask(group_1, group_2_same, timestamps)[1]
    assert not driver_consensus_mask(group_1, group_2_opposite, timestamps)[1]


def test_smooth_group_3_preserves_shape_and_capacity_bounds() -> None:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"]
    )
    group_1 = np.asarray([0.0, 10_000.0, 0.0])
    group_2 = np.asarray([0.0, 8_000.0, 0.0])
    group_3 = np.asarray([0.0, 20_000.0, 0.0])

    result, mask = smooth_group_3(
        group_1, group_2, group_3, timestamps, alpha=0.05, max_delta_ratio=1.0
    )

    assert result.shape == group_3.shape
    assert mask.shape == group_3.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 21_000.0)
    assert result[1] < group_3[1]
