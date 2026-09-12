from __future__ import annotations

import numpy as np

from src.archive.v294_abs_regular_sanity_audit import blend


def test_probability_blend_preserves_inactive_rows() -> None:
    parent = np.array([0.2, 0.4, 0.8])
    expert = np.array([0.6, 0.5])
    active = np.array([True, False, True])
    actual = blend(parent, expert, active, 0.1, "probability")
    np.testing.assert_allclose(actual, [0.24, 0.4, 0.77])


def test_logit_blend_is_bounded_and_preserves_inactive_rows() -> None:
    parent = np.array([0.001, 0.4, 0.999])
    expert = np.array([0.9, 0.1])
    active = np.array([True, False, True])
    actual = blend(parent, expert, active, 0.1, "logit")
    assert actual[1] == parent[1]
    assert np.all((actual >= 0.001) & (actual <= 0.999))
