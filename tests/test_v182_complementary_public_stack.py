import numpy as np

from src.archive.v182_complementary_public_stack import additive_stack


def test_additive_stack_preserves_inactive_public_direction() -> None:
    parent = np.array([0.4, 0.5, 0.6])
    direction = np.array([0.04, -0.08, 0.12])
    independent = np.array([0.6, 0.1, 0.2])
    active = np.array([True, False, True])
    result = additive_stack(parent, direction, independent, active, 0.01)
    expected = parent + 0.25 * direction
    expected[active] += 0.01 * (independent - parent)[active]
    np.testing.assert_allclose(result, expected)
