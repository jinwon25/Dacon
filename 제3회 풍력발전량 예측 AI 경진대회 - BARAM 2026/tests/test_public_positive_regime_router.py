import numpy as np
import pytest

from experiments.public_positive_regime_router import (
    apply_router,
    regime_mask,
)


def test_regime_mask_combines_power_movement_and_direction() -> None:
    result = regime_mask(
        np.asarray([0.10, 0.30, 0.60, 0.80]),
        np.asarray([-0.003, -0.004, 0.004, -0.02]),
        power_lower=0.20,
        power_upper=0.70,
        absolute_lower=0.0025,
        absolute_upper=0.005,
        direction="down",
    )

    np.testing.assert_array_equal(result, [False, True, False, False])


def test_apply_router_changes_only_group2_masked_rows() -> None:
    anchor = {
        "kpx_group_1": np.asarray([1.0, 2.0]),
        "kpx_group_2": np.asarray([3.0, 4.0]),
        "kpx_group_3": np.asarray([5.0, 6.0]),
    }

    result = apply_router(
        anchor,
        np.asarray([30.0, 40.0]),
        np.asarray([True, False]),
    )

    np.testing.assert_array_equal(result["kpx_group_1"], anchor["kpx_group_1"])
    np.testing.assert_array_equal(result["kpx_group_2"], [30.0, 4.0])
    np.testing.assert_array_equal(result["kpx_group_3"], anchor["kpx_group_3"])


def test_regime_mask_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="unknown movement direction"):
        regime_mask(
            np.asarray([0.5]),
            np.asarray([0.01]),
            power_lower=0.0,
            power_upper=1.0,
            absolute_lower=0.0,
            absolute_upper=1.0,
            direction="sideways",
        )
