import numpy as np

from src.archive.v185_public_component_diagnostic import (
    DIAGNOSTIC_WEIGHT,
    MAIN_WEIGHTS,
    STACK_COEFFICIENTS,
    STACK_INTERCEPT,
    diagnostic_blend,
    linear_public_stack,
)


def test_linear_public_stack_matches_frozen_formula() -> None:
    p2 = np.array([0.4, 0.5])
    p55 = np.array([0.5, 0.6])
    p30 = np.array([0.6, 0.7])
    risks = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    main = np.column_stack([p2, p55, p30]) @ MAIN_WEIGHTS
    expected = np.clip(
        STACK_INTERCEPT
        + np.column_stack([main, risks]) @ STACK_COEFFICIENTS,
        1e-6,
        1 - 1e-6,
    )
    np.testing.assert_allclose(linear_public_stack(p2, p55, p30, risks), expected)


def test_diagnostic_blend_preserves_inactive_rows() -> None:
    base = np.array([0.4, 0.5, 0.6])
    independent = np.array([0.8, 0.1, 0.2])
    active = np.array([True, False, True])
    result = diagnostic_blend(base, independent, active)
    expected = base.copy()
    expected[active] += DIAGNOSTIC_WEIGHT * (independent - base)[active]
    np.testing.assert_allclose(result, expected)
