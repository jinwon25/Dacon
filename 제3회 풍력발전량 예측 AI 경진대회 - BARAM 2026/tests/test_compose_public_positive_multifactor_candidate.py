import numpy as np
import pytest

from experiments.compose_public_positive_multifactor_candidate import (
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
