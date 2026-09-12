from __future__ import annotations

import numpy as np

from experiments.compose_public_positive_factor_expansion import (
    expand_factor,
)


def test_expand_factor_recovers_double_weight_and_cap() -> None:
    control = np.array([100.0, 100.0, 100.0])
    treatment = np.array([105.0, 95.0, 140.0])
    expanded = expand_factor(
        control,
        treatment,
        base_weight=0.05,
        expanded_weight=0.10,
        capacity=1_000.0,
        maximum_movement_ratio=0.05,
    )
    assert np.allclose(expanded, [110.0, 90.0, 150.0])


def test_expand_factor_rejects_non_expansion() -> None:
    try:
        expand_factor(
            np.array([1.0]),
            np.array([2.0]),
            base_weight=0.05,
            expanded_weight=0.05,
            capacity=10.0,
            maximum_movement_ratio=0.05,
        )
    except ValueError as error:
        assert "must exceed" in str(error)
    else:
        raise AssertionError("invalid expansion was accepted")
