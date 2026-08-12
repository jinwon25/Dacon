import numpy as np
import pandas as pd
import pytest

import script


def test_scalar_trackman_weight_is_backward_compatible():
    frame = pd.DataFrame({"game_type": ["R", "F", "X"]})
    actual = script.resolve_trackman_weights(frame, {"trackman_weight": 0.05})
    assert np.allclose(actual, [0.05, 0.05, 0.05])


def test_game_type_trackman_weights_are_row_local():
    frame = pd.DataFrame({"game_type": ["R", "F", "X", None]})
    actual = script.resolve_trackman_weights(
        frame,
        {
            "trackman_default_weight": 0.0,
            "trackman_weight_by_game_type": {"R": 0.10, "F": 1.0},
        },
    )
    assert np.allclose(actual, [0.10, 1.0, 0.0, 0.0])


def test_game_type_trackman_weights_reject_invalid_values():
    frame = pd.DataFrame({"game_type": ["R"]})
    with pytest.raises(ValueError):
        script.resolve_trackman_weights(
            frame,
            {"trackman_weight_by_game_type": {"R": 1.1}},
        )
