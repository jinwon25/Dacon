from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v92_temporal_player_command_eb import (
    apply_probability_correction,
    fit_eb_effect,
    group_keys,
    map_effect,
    sign_consensus_effect,
)


def test_sign_consensus_keeps_only_shared_same_sign() -> None:
    output = sign_consensus_effect(
        {"a": 0.04, "b": -0.09, "c": 0.01},
        {"a": 0.01, "b": 0.04, "d": -0.03},
    )
    assert output == {"a": 0.02}


def test_unknown_group_maps_to_zero() -> None:
    mapped = map_effect(np.array(["known", "unknown"]), {"known": 0.02})
    np.testing.assert_allclose(mapped, [0.02, 0.0])


def test_fit_and_map_are_row_order_independent() -> None:
    keys = np.array(["a", "a", "b", "b"])
    target = np.array([1.0, 0.0, 1.0, 1.0])
    parent = np.full(4, 0.5)
    active = np.ones(4, dtype=bool)
    effect = fit_eb_effect(keys, target, parent, active, alpha=2.0)
    original = map_effect(keys, effect)
    order = np.array([2, 0, 3, 1])
    shuffled = map_effect(keys[order], effect)
    np.testing.assert_allclose(shuffled, original[order])


def test_probability_correction_changes_only_route() -> None:
    frame = pd.DataFrame({"domain3": ["R_CORE", "F", "R_ANCHOR"]})
    parent = np.array([0.5, 0.5, 0.5])
    output, active = apply_probability_correction(
        frame, parent, np.array([0.02, 0.02, -0.02]), eta=0.5
    )
    np.testing.assert_array_equal(active, [True, False, False])
    np.testing.assert_allclose(output, [0.51, 0.5, 0.5])


def test_group_keys_depend_only_on_each_row() -> None:
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    original = group_keys(frame, ("a", "b"))
    shuffled = group_keys(frame.iloc[::-1].reset_index(drop=True), ("a", "b"))
    np.testing.assert_array_equal(shuffled, original[::-1])
