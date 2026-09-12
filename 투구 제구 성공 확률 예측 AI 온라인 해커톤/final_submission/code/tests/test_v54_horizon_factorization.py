from __future__ import annotations

import numpy as np
import pytest

from src.archive.v54_horizon_factorization import combine_horizons


def test_mean_horizon_is_equal_weight_average() -> None:
    older = np.array([0.2, -0.4, 0.0])
    latest = np.array([0.4, 0.2, -0.2])
    assert np.allclose(
        combine_horizons(older, latest, "mean"),
        np.array([0.3, -0.1, -0.1]),
    )


def test_agreement_horizon_zeros_sign_conflicts() -> None:
    older = np.array([0.2, -0.4, 0.0, -0.2])
    latest = np.array([0.4, 0.2, -0.2, -0.6])
    assert np.allclose(
        combine_horizons(older, latest, "agree"),
        np.array([0.3, 0.0, 0.0, -0.4]),
    )


def test_horizon_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        combine_horizons(np.zeros(2), np.zeros(3), "mean")
    with pytest.raises(ValueError):
        combine_horizons(np.zeros(2), np.zeros(2), "latest")
