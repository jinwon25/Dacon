from __future__ import annotations

import numpy as np

from experiments.ldaps_upwind_advection_power_curve import (
    CAPACITY,
    AdvectionPolicy,
    apply_advection_policy,
    upwind_kernel_speed,
)


def test_upwind_kernel_follows_incoming_wind_direction() -> None:
    # Positive u means flow to the east, so the physically upwind point is west.
    hub_u = np.asarray([[10.0, 2.0]])
    hub_v = np.zeros_like(hub_u)
    grid_x = np.asarray([-2.0, 2.0])
    grid_y = np.asarray([0.0, 0.0])
    center = upwind_kernel_speed(
        hub_u, hub_v, grid_x, grid_y, 0.0, sigma_km=1.0
    )
    upstream = upwind_kernel_speed(
        hub_u, hub_v, grid_x, grid_y, 2.0, sigma_km=1.0
    )
    assert upstream[0] > center[0]
    assert upstream[0] > 9.9


def test_advection_policy_bounds_increment_and_direction() -> None:
    incumbent = np.asarray([5_000.0, 5_000.0, 5_000.0])
    member = np.asarray([9_000.0, 1_000.0, 9_000.0])
    policy = AdvectionPolicy(
        upwind_distance_km=3.0,
        direction="up",
        coverage=0.10,
        minimum_base_ratio=0.10,
        maximum_base_ratio=0.80,
        minimum_disagreement_kwh=1_000.0,
        alpha=1.0,
    )
    candidate, gate = apply_advection_policy(
        member,
        incumbent,
        np.asarray([True, True, False]),
        policy,
        maximum_incremental_movement_ratio=0.01,
    )
    np.testing.assert_array_equal(gate, [True, False, False])
    np.testing.assert_allclose(
        candidate,
        [5_000.0 + 0.01 * CAPACITY, 5_000.0, 5_000.0],
    )
