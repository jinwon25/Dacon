import numpy as np
import pytest

from src.archive.v256_pitchmix_release_fixed_audit import mix_pitchmix


def test_mix_pitchmix_uses_one_fixed_convex_dose() -> None:
    base = np.array([0.2, 0.5, 0.8])
    pitchmix = np.array([0.4, 0.4, 0.6])
    np.testing.assert_allclose(mix_pitchmix(base, pitchmix), [0.25, 0.475, 0.75])


def test_mix_pitchmix_rejects_shape_or_weight_errors() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        mix_pitchmix(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_pitchmix(np.zeros(2), np.zeros(2), weight=1.1)
