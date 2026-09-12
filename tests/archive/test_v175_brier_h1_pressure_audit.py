from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v175_brier_h1_pressure_audit import pressure_mask, route_candidate


def test_pressure_mask_and_route_are_row_local() -> None:
    frame = pd.DataFrame({
        "balls_before": [0, 3, 1, 2],
        "strikes_before": [2, 0, 1, 1],
    })
    np.testing.assert_array_equal(pressure_mask(frame), [True, True, False, False])
    axis = {
        "exact_mask": np.array([True, True, True, False]),
        "domain3": np.array(["R_CORE", "F", "R_CORE", "R_CORE"]),
    }
    base = np.array([0.4, 0.4, 0.4, 0.4])
    alt = np.array([0.6, 0.6, 0.6, 0.6])
    candidate, active = route_candidate(base, alt, axis, frame)
    np.testing.assert_array_equal(active, [True, False, False, False])
    np.testing.assert_allclose(candidate, [0.6, 0.4, 0.4, 0.4])
