from __future__ import annotations

import numpy as np
import pytest

from experiments.compose_public_positive_g3_expansion import (
    expand_group3_factor,
)


def test_expand_group3_factor_scales_observed_difference() -> None:
    control = np.asarray([1_000.0, 5_000.0, 9_000.0])
    treatment = np.asarray([1_100.0, 4_800.0, 9_400.0])
    expanded = expand_group3_factor(
        control,
        treatment,
        multiplier=1.25,
        capacity=10_000.0,
        maximum_factor_ratio=0.10,
    )
    assert expanded.tolist() == pytest.approx([1_125.0, 4_750.0, 9_500.0])


def test_expand_group3_factor_caps_and_clips() -> None:
    expanded = expand_group3_factor(
        np.asarray([100.0, 9_900.0]),
        np.asarray([-5_000.0, 20_000.0]),
        multiplier=2.0,
        capacity=10_000.0,
        maximum_factor_ratio=0.10,
    )
    assert expanded.tolist() == pytest.approx([0.0, 10_000.0])


def test_expand_group3_factor_rejects_unsafe_arguments() -> None:
    values = np.asarray([1.0, 2.0])
    with pytest.raises(ValueError, match="at least"):
        expand_group3_factor(
            values,
            values,
            multiplier=0.9,
            capacity=10.0,
            maximum_factor_ratio=0.1,
        )
    with pytest.raises(ValueError, match="align"):
        expand_group3_factor(
            values,
            np.asarray([1.0]),
            multiplier=1.1,
            capacity=10.0,
            maximum_factor_ratio=0.1,
        )
