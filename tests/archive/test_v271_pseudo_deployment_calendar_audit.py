import numpy as np

from src.archive.v271_pseudo_deployment_calendar_audit import (
    calendar_equal_fallback,
)


def test_calendar_equal_fallback_changes_only_april_through_september_active() -> None:
    base = np.array([0.4, 0.4, 0.4, 0.4])
    pseudo = np.array([0.6, 0.6, 0.6, 0.6])
    active = np.array([True, True, False, True])
    month = np.array([3, 4, 7, 10])
    fallback, selected = calendar_equal_fallback(base, pseudo, active, month)
    np.testing.assert_allclose(fallback, [0.4, 0.5, 0.4, 0.4])
    np.testing.assert_array_equal(selected, [False, True, False, False])
