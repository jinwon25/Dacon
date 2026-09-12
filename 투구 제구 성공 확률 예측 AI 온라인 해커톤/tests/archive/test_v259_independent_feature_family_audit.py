import numpy as np
import pytest

from src.archive.v259_independent_feature_family_audit import (
    SUBSETS,
    blend_feature_models,
)


def test_family_is_preregistered_as_four_singles_and_six_pairs() -> None:
    assert len(SUBSETS) == 10
    assert sum(len(subset) == 1 for subset in SUBSETS) == 4
    assert sum(len(subset) == 2 for subset in SUBSETS) == 6


def test_blend_feature_models_has_one_fixed_total_dose() -> None:
    base = np.array([0.2, 0.8])
    left = np.array([0.4, 0.6])
    right = np.array([0.6, 0.4])
    np.testing.assert_allclose(
        blend_feature_models(base, [left, right]), [0.275, 0.725]
    )


def test_blend_feature_models_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        blend_feature_models(np.zeros(2), [])
    with pytest.raises(ValueError, match="different shapes"):
        blend_feature_models(np.zeros(2), [np.zeros(3)])
