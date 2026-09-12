from __future__ import annotations

import numpy as np

from src.archive.v170_current_residual_catboost import apply_correction


def test_apply_correction_is_row_local_clipped_and_preserves_inactive() -> None:
    base = np.array([0.20, 0.40, 0.98])
    correction = np.array([0.20, -0.02, 0.20])
    active = np.array([False, True, True])
    output = apply_correction(base, correction, active, dose=1.0)
    np.testing.assert_allclose(output, np.array([0.20, 0.38, 0.999]))
