from __future__ import annotations

import numpy as np
import pytest

from src.archive.v172_h1_seed_subset_audit import seed_subsets, subset_mean


def test_seed_subsets_are_all_seven_nonempty_combinations() -> None:
    subsets = seed_subsets()
    assert len(subsets) == 7
    assert (42,) in subsets
    assert (42, 43, 44) in subsets
    assert len(set(subsets)) == 7


def test_subset_mean_uses_only_requested_seed_predictions() -> None:
    values = {
        42: np.array([0.1, 0.2]),
        43: np.array([0.3, 0.6]),
        44: np.array([0.9, 1.0]),
    }
    np.testing.assert_allclose(subset_mean(values, (42, 43)), [0.2, 0.4])
    with pytest.raises(ValueError, match="invalid seed subset"):
        subset_mean(values, ())
