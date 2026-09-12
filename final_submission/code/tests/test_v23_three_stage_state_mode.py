import numpy as np

from src.archive.v23_three_stage_state_mode_screen import _diagnostics


def test_diagnostics_prefers_perfect_signal():
    target = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    parent = np.full(6, 0.5)
    month = np.array([3, 3, 4, 4, 5, 5])
    domain = np.array(["R_CORE", "R_ANCHOR", "F", "R_CORE", "R_ANCHOR", "F"])
    result = _diagnostics(target, parent, target, month, domain, 0.5)
    assert result["gain"] > 0
    assert result["worst_month_gain"] > 0
    assert result["minimum_domain_gain"] > 0
    assert result["selection_score"] > 0
