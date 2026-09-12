import numpy as np

from src.archive.v191_leave_one_year_minimax_stack import (
    form_gain,
    quadratic_gain_form,
    stack_prediction,
)


def test_quadratic_form_matches_direct_bss_gain() -> None:
    target = np.array([0.0, 1.0, 1.0, 0.0])
    base = np.array([0.4, 0.6, 0.5, 0.3])
    direction = np.array([[0.1], [-0.1], [0.05], [-0.05]])
    weight = np.array([0.2])
    mask = np.ones(4, dtype=bool)
    candidate = base + direction[:, 0] * weight[0]
    rate = target.mean()
    direct = 100_000.0 * (
        np.mean((target - base) ** 2) - np.mean((target - candidate) ** 2)
    ) / (rate * (1.0 - rate))
    assert abs(form_gain(weight, quadratic_gain_form(target, base, direction, mask)) - direct) < 1e-10


def test_stack_prediction_clips_bounds() -> None:
    parent = np.array([0.99, 0.01])
    direction = np.array([[1.0], [-1.0]])
    result = stack_prediction(parent, direction, np.array([0.1]))
    np.testing.assert_allclose(result, [0.999, 0.001])
