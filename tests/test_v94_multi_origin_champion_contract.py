from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v94_multi_origin_champion_contract import (
    domain3,
    sha256_array,
    sha256_strings,
    validate_axis,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 13, 1],
            "batter_team_id": [2, 2, 2],
            "control_success": [1, 0, 1],
        }
    )


def test_domain3_is_row_local() -> None:
    frame = _frame()
    np.testing.assert_array_equal(domain3(frame), ["R_CORE", "R_ANCHOR", "F"])
    order = np.array([2, 0, 1])
    np.testing.assert_array_equal(domain3(frame.iloc[order]), domain3(frame)[order])


def test_array_digest_depends_on_dtype_shape_and_values() -> None:
    values = np.array([1, 2, 3], dtype=np.int64)
    assert sha256_array(values) == sha256_array(values.copy())
    assert sha256_array(values) != sha256_array(values.astype(np.int32))
    assert sha256_array(values) != sha256_array(values[::-1])


def test_string_digest_is_order_sensitive() -> None:
    values = pd.Series(["a", "b", "가"])
    assert sha256_strings(values) == sha256_strings(values.copy())
    assert sha256_strings(values) != sha256_strings(values.iloc[::-1])


def test_validate_axis_accepts_aligned_binary_probability() -> None:
    frame = _frame()
    validate_axis(frame, np.array([10, 11, 12]), np.array([1, 0, 1]), np.array([0.8, 0.2, 0.7]))


def test_validate_axis_rejects_duplicate_index_or_bad_probability() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="unique"):
        validate_axis(frame, np.array([10, 10, 12]), np.array([1, 0, 1]), np.array([0.8, 0.2, 0.7]))
    with pytest.raises(ValueError, match="probability"):
        validate_axis(frame, np.array([10, 11, 12]), np.array([1, 0, 1]), np.array([0.8, 1.2, 0.7]))
