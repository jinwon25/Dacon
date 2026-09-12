import numpy as np
import pandas as pd

from src.archive.v332_futures_reliability_dose_router import (
    PROTOCOL,
    apply_router,
    sse_gain,
)


def test_v332_protocol() -> None:
    assert PROTOCOL == "V332_FUTURES_RELIABILITY_DOSE_ROUTER_V1"


def test_sse_gain_rewards_correct_direction() -> None:
    target = np.asarray([1.0, 0.0])
    parent = np.asarray([0.4, 0.6])
    step = np.asarray([0.05, -0.05])
    mask = np.ones(2, dtype=bool)
    assert sse_gain(target, parent, step, mask, 1) > 0.0
    assert sse_gain(target, parent, step, mask, -1) < 0.0


def test_apply_router_changes_only_futures_matched_cells() -> None:
    parent = np.asarray([0.4, 0.5, 0.6])
    step = np.asarray([0.01, 0.02, -0.01])
    keys = np.asarray(["A", "A", "B"])
    futures = np.asarray([True, False, True])
    candidate, active, action = apply_router(parent, step, keys, {"A": 1}, futures)
    np.testing.assert_allclose(candidate, [0.41, 0.5, 0.6])
    np.testing.assert_array_equal(active, [True, False, False])
    np.testing.assert_array_equal(action, [1, 0, 0])
