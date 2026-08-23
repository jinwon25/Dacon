from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v85_lowrank_policy_replacement import (
    exact_v84_parent,
    replace_strict_policy,
)


def test_exact_v84_parent_replaces_only_f() -> None:
    axis = {
        "parent": np.array([0.4, 0.5, 0.6]),
        "domain3": np.array(["R_CORE", "F", "R_ANCHOR"]),
    }
    v82 = np.array([0.41, 0.5, 0.59])
    v56 = np.array([0.4, 0.53, 0.6])
    output, audit = exact_v84_parent(axis, v82, v56)
    assert np.allclose(output, [0.41, 0.53, 0.59])
    assert audit["f_rows"] == 1
    assert audit["v82_f_parent_max_abs"] == 0.0


def test_exact_v84_parent_rejects_overlapping_f_change() -> None:
    axis = {
        "parent": np.array([0.4, 0.5]),
        "domain3": np.array(["R_CORE", "F"]),
    }
    with pytest.raises(ValueError, match="v82 unexpectedly changed F"):
        exact_v84_parent(axis, np.array([0.4, 0.51]), np.array([0.4, 0.52]))


def test_replace_strict_policy_is_row_local_and_route_limited() -> None:
    frame = pd.DataFrame(
        {"domain3": ["R_CORE", "F", "R_ANCHOR", "R_CORE"]}
    )
    parent = np.array([0.4, 0.5, 0.6, 0.7])
    equal = np.array([0.3, 0.4, 0.5, 0.6])
    policy = np.array([0.5, 0.9, 0.1, 0.4])
    output, active = replace_strict_policy(frame, parent, equal, policy)
    assert np.array_equal(active, [True, False, False, True])
    assert np.allclose(output, [0.42, 0.5, 0.6, 0.68])

    permutation = np.array([3, 1, 0, 2])
    shuffled, _ = replace_strict_policy(
        frame.iloc[permutation].reset_index(drop=True),
        parent[permutation],
        equal[permutation],
        policy[permutation],
    )
    assert np.allclose(shuffled, output[permutation])
