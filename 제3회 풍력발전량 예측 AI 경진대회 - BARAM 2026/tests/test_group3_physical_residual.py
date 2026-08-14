from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.group3_physical_residual import (
    CAPACITY,
    Policy,
    _air_density,
    _shear,
    _veer,
    apply_policy,
)


def test_moist_air_density_is_physical_and_humidity_reduces_density() -> None:
    index = pd.RangeIndex(2)
    pressure = pd.Series([101_325.0, 101_325.0], index=index)
    temperature = pd.Series([288.15, 288.15], index=index)
    humidity = pd.Series([0.0, 100.0], index=index)

    density = _air_density(pressure, temperature, humidity)

    assert np.all((density > 1.1) & (density < 1.3))
    assert density.iloc[1] < density.iloc[0]


def test_shear_and_veer_have_bounded_physical_outputs() -> None:
    lower = pd.Series([5.0, 10.0, 0.05])
    upper = pd.Series([10.0, 5.0, 50.0])
    shear = _shear(lower, upper, 10.0, 50.0)
    veer = _veer(
        pd.Series([1.0, 1.0]),
        pd.Series([0.0, 0.0]),
        pd.Series([0.0, 1.0]),
        pd.Series([1.0, 0.0]),
    )

    assert np.all((shear >= -0.30) & (shear <= 0.60))
    assert np.isclose(veer.iloc[0], np.pi / 2.0)
    assert np.isclose(veer.iloc[1], 0.0)


def test_bounded_policy_requires_seed_consensus_and_limits_coverage() -> None:
    current = np.full(4, 0.4 * CAPACITY)
    members = np.array(
        [
            [0.42 * CAPACITY, 0.43 * CAPACITY, 0.42 * CAPACITY],
            [0.42 * CAPACITY, 0.38 * CAPACITY, 0.41 * CAPACITY],
            [0.50 * CAPACITY, 0.50 * CAPACITY, 0.50 * CAPACITY],
            [0.405 * CAPACITY, 0.405 * CAPACITY, 0.405 * CAPACITY],
        ]
    )
    policy = Policy(
        family="test",
        alpha=0.10,
        min_disagreement_ratio=0.01,
        max_disagreement_ratio=0.06,
        max_seed_std_ratio=0.01,
    )

    candidate, gate = apply_policy(current, members, policy)

    assert gate.tolist() == [True, False, False, False]
    assert candidate[0] > current[0]
    assert np.array_equal(candidate[1:], current[1:])
