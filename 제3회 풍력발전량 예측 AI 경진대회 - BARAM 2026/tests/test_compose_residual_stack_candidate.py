from __future__ import annotations

import numpy as np
import pytest

from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)


def test_apply_capped_residual_stack_adds_fraction_and_caps_total() -> None:
    reference = np.asarray([50.0, 50.0, 50.0])
    primary = np.asarray([54.0, 46.0, 50.0])
    residual = np.asarray([58.0, 42.0, 52.0])

    result = apply_capped_residual_stack(
        reference,
        primary,
        residual,
        residual_weight=0.5,
        capacity=100.0,
        movement_cap_ratio=0.05,
    )

    np.testing.assert_allclose(result, [55.0, 45.0, 51.0])


def test_apply_capped_residual_stack_rejects_bad_weight() -> None:
    values = np.ones(2)
    with pytest.raises(ValueError, match="residual weight"):
        apply_capped_residual_stack(
            values,
            values,
            values,
            residual_weight=1.1,
            capacity=100.0,
            movement_cap_ratio=0.05,
        )
