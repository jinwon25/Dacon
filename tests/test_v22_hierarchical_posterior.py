from __future__ import annotations

import numpy as np

from src.v22_hierarchical_posterior_screen import _posterior, _rounded_count


def test_rounded_count_and_posterior() -> None:
    count = _rounded_count(
        np.asarray([0.0, 10.0, 12.0]), np.asarray([np.nan, 0.3, 0.5])
    )
    assert np.array_equal(count, [0.0, 3.0, 6.0])
    state = {"season_s": np.asarray([0.0, 3.0]), "season_n": np.asarray([0.0, 5.0])}
    result = _posterior(state, np.asarray([0.4, 0.6]), 10.0)
    assert np.allclose(result, [0.4, 0.6])
