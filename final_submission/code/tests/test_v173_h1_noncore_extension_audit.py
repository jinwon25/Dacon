from __future__ import annotations

import numpy as np

from src.archive.v173_h1_noncore_extension_audit import extend_h1, paired_metrics


def test_extend_h1_changes_only_exact_selected_domain() -> None:
    base = np.array([0.4, 0.5, 0.6, 0.7])
    component = base.copy()
    h1 = np.array([0.8, 0.1, 0.2, 0.3])
    domain = np.array(["F", "R_CORE", "F", "F"])
    exact = np.array([True, True, False, True])
    candidate, active = extend_h1(
        base, component, h1, domain, exact, ("F",), 0.25
    )
    np.testing.assert_array_equal(active, [True, False, False, True])
    np.testing.assert_allclose(candidate, [0.5, 0.5, 0.6, 0.6])


def test_paired_metrics_reports_positive_gain() -> None:
    axis = {
        "exact_mask": np.ones(4, dtype=bool),
        "target": np.array([1.0, 0.0, 1.0, 0.0]),
        "game_month": np.array([4, 4, 5, 5]),
    }
    base = np.full(4, 0.5)
    candidate = np.array([0.6, 0.4, 0.6, 0.4])
    result = paired_metrics(axis, base, candidate, np.ones(4, dtype=bool))
    assert result["overall_gain"] > 0.0
    assert result["active_domain_gain"] > 0.0
    assert result["positive_month_fraction"] == 1.0
