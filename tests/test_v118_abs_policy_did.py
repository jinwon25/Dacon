import numpy as np
import pandas as pd

from src.archive.v118_abs_policy_did import (
    difference_in_differences,
    map_effect,
)


def _frame(values):
    return pd.DataFrame({"count_state": values})


def test_difference_in_differences_removes_common_change():
    pre = _frame(["a", "a", "b", "b", "a", "a", "b", "b"])
    post = pre.copy()
    treated = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    control = ~treated
    pre_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    post_y = pre_y.copy()
    post_y[:2] += 0.4
    effect = difference_in_differences(
        pre,
        pre_y,
        post,
        post_y,
        ("count_state",),
        treated_pre=treated,
        control_pre=control,
        treated_post=treated,
        control_post=control,
        alpha=0.0,
        minimum_rows=1,
    )
    assert effect["a"] > 0.0
    assert effect["b"] < 0.0


def test_map_effect_has_zero_backoff_and_cap():
    frame = _frame(["a", "unknown"])
    result = map_effect(frame, ("count_state",), {"a": 0.5}, 0.03)
    np.testing.assert_allclose(result, [0.03, 0.0])
