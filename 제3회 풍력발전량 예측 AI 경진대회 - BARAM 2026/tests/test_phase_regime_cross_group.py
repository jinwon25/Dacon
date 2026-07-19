from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.phase_regime_cross_group import (
    blend_components,
    inject_member,
    lead_phase,
)
from src.metrics import CAPACITY_KWH


def test_lead_phase_resets_at_issue_cycle_boundary() -> None:
    index = pd.DatetimeIndex(
        ["2025-01-01 01:00:00", "2025-01-01 06:00:00", "2025-01-01 07:00:00", "2025-01-02 00:00:00"]
    )
    assert lead_phase(index).tolist() == [0, 0, 1, 3]


def test_component_blend_is_convex_average() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = blend_components(matrix, (0.25, 0.50, 0.25))
    np.testing.assert_allclose(result, [2.0, 5.0])


def test_injection_respects_disagreement_and_driver_gate() -> None:
    capacity = CAPACITY_KWH["kpx_group_3"]
    base = np.array([0.20, 0.20, 0.05]) * capacity
    member = np.array([0.22, 0.26, 0.06]) * capacity
    group_1 = np.array([0.30, 0.20, 0.30]) * CAPACITY_KWH["kpx_group_1"]
    group_2 = np.array([0.31, 0.40, 0.31]) * CAPACITY_KWH["kpx_group_2"]
    result, gate = inject_member(
        base,
        member,
        group_1,
        group_2,
        alpha=0.25,
        max_disagreement=0.04,
        require_driver_agreement=True,
    )
    assert gate.tolist() == [True, False, False]
    assert result[0] == base[0] + 0.25 * (member[0] - base[0])
    np.testing.assert_allclose(result[1:], base[1:])
