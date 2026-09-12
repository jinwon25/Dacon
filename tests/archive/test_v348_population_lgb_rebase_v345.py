from __future__ import annotations

import numpy as np

from src.archive.v348_population_lgb_rebase_v345 import apply_frozen_direction


def test_apply_frozen_direction_changes_only_active_rows() -> None:
    parent = np.array([0.2, 0.5, 0.8])
    direction = np.array([0.1, -0.2, 0.3])
    active = np.array([True, False, True])
    actual = apply_frozen_direction(parent, direction, active)
    np.testing.assert_allclose(actual, np.array([0.3, 0.5, 0.999]))


def test_apply_frozen_direction_is_row_local() -> None:
    parent = np.array([0.2, 0.5, 0.8])
    direction = np.array([0.1, -0.2, 0.05])
    active = np.array([True, False, True])
    order = np.array([2, 0, 1])
    expected = apply_frozen_direction(parent, direction, active)[order]
    actual = apply_frozen_direction(parent[order], direction[order], active[order])
    np.testing.assert_allclose(actual, expected)
