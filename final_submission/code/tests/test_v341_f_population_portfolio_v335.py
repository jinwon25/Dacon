import numpy as np
import pytest

from src.archive import v341_f_population_portfolio_v335 as v341


def test_add_frozen_increment_changes_only_active_rows() -> None:
    result = v341.add_frozen_increment(
        np.array([0.2, 0.4, 0.6]),
        np.array([0.1, 0.3, 0.7]),
        np.array([0.15, 0.2, 0.8]),
        np.array([True, False, True]),
    )
    assert np.allclose(result, [0.25, 0.4, 0.7])


def test_add_frozen_increment_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        v341.add_frozen_increment(
            np.array([0.2]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            np.array([True, False]),
        )
