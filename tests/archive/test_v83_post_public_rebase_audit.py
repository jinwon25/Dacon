from __future__ import annotations

import numpy as np
import pytest

from src.archive.v83_post_public_rebase_audit import add_frozen_shift


def test_add_frozen_shift_preserves_declared_direction() -> None:
    old = np.array([0.2, 0.5, 0.8])
    new = np.array([0.3, 0.4, 0.7])
    candidate = np.array([0.25, 0.45, 0.9])
    assert add_frozen_shift(new, old, candidate).tolist() == pytest.approx(
        [0.35, 0.35, 0.8]
    )


def test_add_frozen_shift_clips_and_validates() -> None:
    old = np.array([0.5, 0.5])
    new = np.array([0.99, 0.01])
    candidate = np.array([0.9, 0.1])
    assert add_frozen_shift(new, old, candidate).tolist() == pytest.approx(
        [0.999, 0.001]
    )
    with pytest.raises(ValueError):
        add_frozen_shift(new, old[:1], candidate)
