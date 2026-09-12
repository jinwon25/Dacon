import numpy as np

from src.archive import v340_calendar_portfolio_v335 as v340


def test_add_component_is_additive_only_on_active_rows() -> None:
    result = v340.add_component(
        np.array([0.2, 0.4]),
        np.array([0.1, 0.3]),
        np.array([0.15, 0.2]),
        np.array([True, False]),
    )
    assert np.allclose(result, [0.25, 0.4])
