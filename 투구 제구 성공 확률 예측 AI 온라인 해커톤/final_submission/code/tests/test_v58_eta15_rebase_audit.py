from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v58_eta15_rebase_audit import eta_parent, rebase_frozen_shift


def test_eta_parent_recovers_same_signal_weight() -> None:
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.6],
            "v25": [0.415, 0.585],
        }
    )

    parent = eta_parent(frame, 0.15)

    np.testing.assert_allclose(parent, [0.43, 0.57], rtol=0.0, atol=1e-15)


def test_rebase_preserves_frozen_probability_shift() -> None:
    old_parent = np.array([0.4, 0.5])
    old_candidate = np.array([0.41, 0.48])
    current_parent = np.array([0.45, 0.55])

    rebased, shift = rebase_frozen_shift(
        old_parent, old_candidate, current_parent
    )

    np.testing.assert_allclose(shift, [0.01, -0.02], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(rebased, [0.46, 0.53], rtol=0.0, atol=1e-15)


def test_rebase_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="aligned"):
        rebase_frozen_shift(
            np.array([0.4]), np.array([0.4, 0.5]), np.array([0.4])
        )
