import numpy as np
import pytest

from src.archive.v253_fixed_route_command_dispersion_audit import mix_fallback


def test_fixed_dispersion_mix_is_convex() -> None:
    base = np.array([0.2, 0.8])
    command = np.array([0.6, 0.4])
    np.testing.assert_allclose(mix_fallback(base, command), [0.3, 0.7])


def test_fixed_dispersion_mix_validates_contract() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        mix_fallback(np.array([0.2]), np.array([0.2, 0.3]))
    with pytest.raises(ValueError, match="weight"):
        mix_fallback(np.array([0.2]), np.array([0.3]), 1.1)
