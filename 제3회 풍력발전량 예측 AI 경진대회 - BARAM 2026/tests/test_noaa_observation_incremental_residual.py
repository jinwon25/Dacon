from __future__ import annotations

import numpy as np

from experiments.noaa_observation_incremental_residual import (
    FROZEN_TARGET,
    Policy,
    compose_increment,
)
from src.metrics import CAPACITY_KWH


def test_compose_increment_is_bounded_and_freezes_group3() -> None:
    active = {
        target: np.full(4, 0.5 * capacity)
        for target, capacity in CAPACITY_KWH.items()
    }
    increment = {
        "kpx_group_1": np.array([0.02, -0.02, 0.001, -0.001]),
        "kpx_group_2": np.array([0.02, -0.02, 0.001, -0.001]),
    }
    policy = Policy(alpha=0.5, cap_ratio=0.005)

    output = compose_increment(active, increment, policy)

    expected = 0.5 * 0.005 * CAPACITY_KWH["kpx_group_1"]
    assert output["kpx_group_1"][0] == active["kpx_group_1"][0] + expected
    assert output["kpx_group_1"][1] == active["kpx_group_1"][1] - expected
    assert np.array_equal(output[FROZEN_TARGET], active[FROZEN_TARGET])
