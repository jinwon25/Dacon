from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.issue_residual_analog_scenarios import (
    ACTION_RATIOS,
    Policy,
    _expected_actions,
    _rank_pool_neighbors,
    build_issue_trajectories,
    issue_row_mask,
)


def _toy_trajectories():
    index = pd.date_range("2024-01-01 01:00:00", periods=48, freq="h")
    issue = pd.DatetimeIndex(
        [pd.Timestamp("2023-12-31 13:00:00")] * 24
        + [pd.Timestamp("2024-01-01 13:00:00")] * 24
    )
    base_feature = np.arange(48, dtype=float)[:, None]
    views = {
        "power": base_feature,
        "ldaps_power": np.column_stack([base_feature, base_feature + 1.0]),
        "gfs_power": np.column_stack([base_feature, base_feature + 2.0]),
    }
    truth = np.linspace(3_000.0, 6_000.0, 48)
    base = truth - 100.0
    return index, issue, views, truth, base


def test_build_issue_trajectories_preserves_complete_paths():
    index, issue, views, truth, base = _toy_trajectories()
    trajectories = build_issue_trajectories(
        index, issue, views, truth, base
    )
    assert len(trajectories.issues) == 2
    assert trajectories.residual_ratio.shape == (2, 24)
    assert trajectories.views["power"].shape == (2, 24, 1)
    assert np.array_equal(trajectories.rows[0], np.arange(24))
    rows = issue_row_mask(trajectories, np.array([False, True]), len(index))
    assert np.array_equal(np.flatnonzero(rows), np.arange(24, 48))


def test_rank_pool_rejects_query_in_archive():
    index, issue, views, truth, base = _toy_trajectories()
    trajectories = build_issue_trajectories(
        index, issue, views, truth, base
    )
    scales = {
        name: np.ones(values.shape[2], dtype=float)
        for name, values in trajectories.views.items()
    }
    with pytest.raises(ValueError, match="cannot occur"):
        _rank_pool_neighbors(
            trajectories,
            np.array([0, 1]),
            query_position=1,
            neighbors=1,
            scales=scales,
        )


def test_expected_action_keeps_zero_for_symmetric_residuals():
    base = np.full(24, 8_000.0)
    residuals = np.vstack(
        [
            np.full(24, -0.01),
            np.full(24, 0.01),
        ]
    )
    action, advantage = _expected_actions(
        base,
        residuals,
        np.array([0.5, 0.5]),
        eligible_probability=1.0,
        eligible_generation=8_000.0,
    )
    assert np.isin(action, ACTION_RATIOS).all()
    assert np.allclose(action, 0.0)
    assert np.allclose(advantage, 0.0)


def test_policy_name_is_stable():
    assert Policy(12, 0.5, 0.25, 0.65).name == "k12_a50_c25_p65"
