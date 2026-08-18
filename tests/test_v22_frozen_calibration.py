from __future__ import annotations

import numpy as np

from src.v22_frozen_calibration_screen import _apply_map, _bin, _fit_map


def test_fixed_bins_and_frozen_map() -> None:
    prediction = np.asarray([0.30, 0.39, 0.50, 0.70])
    assert _bin(prediction, 4).tolist() == [0, 0, 2, 3]
    target = np.asarray([0.0, 1.0, 1.0, 0.0])
    domain = np.asarray(["R_CORE"] * 4)
    effect = _fit_map(target, prediction, domain, 4, 10.0)
    candidate = _apply_map(prediction, domain, 4, effect, 0.5)
    assert np.isfinite(candidate).all()
    assert np.all((candidate > 0.0) & (candidate < 1.0))
