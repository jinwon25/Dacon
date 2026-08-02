import numpy as np
import pytest

from experiments.compose_public_positive_multifactor_candidate import (
    PUBLIC_INCUMBENT,
    PUBLIC_OBSERVED_EXPANSION,
    _joint_observed_projection,
    expand_observed_factor,
)


def test_expand_observed_factor_scales_from_control() -> None:
    result = expand_observed_factor(
        np.asarray([100.0, 200.0]),
        np.asarray([110.0, 180.0]),
        base_weight=0.05,
        expanded_weight=0.10,
        capacity=1_000.0,
    )

    np.testing.assert_allclose(result, [120.0, 160.0])


def test_expand_observed_factor_caps_large_factor_movement() -> None:
    result = expand_observed_factor(
        np.asarray([100.0]),
        np.asarray([200.0]),
        base_weight=0.05,
        expanded_weight=0.10,
        capacity=1_000.0,
        maximum_factor_ratio=0.05,
    )

    np.testing.assert_allclose(result, [150.0])


def test_expand_observed_factor_rejects_weight_contraction() -> None:
    with pytest.raises(ValueError, match="must not be below"):
        expand_observed_factor(
            np.asarray([100.0]),
            np.asarray([110.0]),
            base_weight=0.10,
            expanded_weight=0.05,
            capacity=1_000.0,
        )


def test_joint_observed_projection_reproduces_calibration_submission() -> None:
    local_delta = {
        "score": 0.00038,
        "one_minus_nmae": 0.000066,
        "ficr": 0.000694,
    }

    result = _joint_observed_projection(local_delta, local_delta)

    for component in local_delta:
        assert result["projected_public_metrics"][component] == pytest.approx(
            PUBLIC_OBSERVED_EXPANSION[component]
        )
        assert result["observed_public_delta"][component] == pytest.approx(
            PUBLIC_OBSERVED_EXPANSION[component]
            - PUBLIC_INCUMBENT[component]
        )
