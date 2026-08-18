import numpy as np
import pandas as pd

from src.v43_independent_base_audit import _compose, diagnostics


def test_compose_changes_only_selected_rows():
    parent = np.array([0.4, 0.5, 0.6])
    direction = np.array([0.2, 0.2, -0.2])
    output = _compose(parent, direction, np.array([True, False, True]), 0.5)
    np.testing.assert_allclose(output, [0.5, 0.5, 0.5])


def test_diagnostics_reports_positive_improvement():
    frame = pd.DataFrame(
        {
            "target": [1.0, 0.0, 1.0, 0.0],
            "game_month": [8, 8, 9, 9],
            "domain3": ["R_CORE", "R_ANCHOR", "F", "R_CORE"],
        }
    )
    parent = np.full(4, 0.5)
    candidate = np.array([0.6, 0.4, 0.6, 0.4])
    result = diagnostics(
        frame, parent, candidate, np.ones(len(frame), dtype=bool)
    )
    assert result["gain"] > 0.0
    assert result["positive_month_fraction"] == 1.0
