import numpy as np

from src.v25_low_dof_failure_three_stage import _conditional_success


def test_conditional_success_uses_recency_weight() -> None:
    target = np.asarray([0.0, 1.0] * 4)
    label = np.repeat(np.arange(4), 2)
    season = np.asarray([2020, 2021] * 4)
    result = _conditional_success(target, label, season, 2022, half_life=1.0)
    np.testing.assert_allclose(result, np.full(4, 2.0 / 3.0))
