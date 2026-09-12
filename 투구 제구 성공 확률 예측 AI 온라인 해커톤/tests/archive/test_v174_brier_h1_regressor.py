from __future__ import annotations

import numpy as np

from src.archive.v174_brier_h1_regressor import mix_h1


def test_mix_h1_endpoints_and_midpoint() -> None:
    current = np.array([0.2, 0.8])
    challenger = np.array([0.6, 0.4])
    np.testing.assert_allclose(mix_h1(current, challenger, 0.0), current)
    np.testing.assert_allclose(mix_h1(current, challenger, 1.0), challenger)
    np.testing.assert_allclose(mix_h1(current, challenger, 0.5), [0.4, 0.6])
