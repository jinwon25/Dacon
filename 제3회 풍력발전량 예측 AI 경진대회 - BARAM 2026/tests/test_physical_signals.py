import numpy as np

from src.physical_signals import (
    density_normalized_wind_speed,
    moist_air_density,
    wake_alignment_index,
)


def test_reference_air_density_leaves_wind_speed_unchanged() -> None:
    speed = density_normalized_wind_speed([10.0], [1.225])
    np.testing.assert_allclose(speed, [10.0])


def test_moist_air_density_is_physical_and_falls_with_humidity() -> None:
    dry, humid = moist_air_density(
        pressure_pa=[101_325.0, 101_325.0],
        temperature_k=[288.15, 288.15],
        specific_humidity=[0.0, 0.02],
    )
    assert 1.20 < dry < 1.25
    assert humid < dry


def test_wake_alignment_is_larger_along_turbine_axis() -> None:
    turbines = ((37.0, 128.0), (37.0, 128.01))
    along = wake_alignment_index([10.0], [0.0], turbines)
    across = wake_alignment_index([0.0], [10.0], turbines)
    assert along[0] > 0.99
    assert across[0] < 1e-6
