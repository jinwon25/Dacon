from __future__ import annotations

import numpy as np

from src.champion.v22_domain_calibration_screen import _candidate


def test_domain_calibration_is_row_local_and_domain_specific() -> None:
    prediction = np.array([0.4, 0.6, 0.3, 0.7])
    domain = np.array(["R_CORE", "R_ANCHOR", "F", "R_CORE"])
    result = _candidate(prediction, domain, 0.5, (0.10, 0.0, 0.20))
    assert np.allclose(result, [0.41, 0.60, 0.34, 0.68])
