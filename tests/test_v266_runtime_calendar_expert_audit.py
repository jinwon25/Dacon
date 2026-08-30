import numpy as np

from src.archive.v266_runtime_calendar_expert_audit import (
    compose_runtime_calendar_experts,
)


def test_runtime_calendar_experts_preserve_inactive_months_and_split_scopes() -> None:
    parent = np.full(4, 0.60)
    base = np.array([0.70, 0.70, 0.70, 0.70])
    command = np.array([0.55, 0.80, 0.55, 0.80])
    batter = np.array([0.40, 0.40, 0.40, 0.40])
    routes = {
        "deployed": np.array([True, True, True, True]),
        "pressure_boundary_agreement": np.zeros(4, dtype=bool),
        "nonpressure_same_hand": np.zeros(4, dtype=bool),
        "nonpressure_opposite_hand_high52": np.zeros(4, dtype=bool),
    }
    month = np.array([4, 4, 3, 10])
    baseline, candidate, selected, command_selected = (
        compose_runtime_calendar_experts(
            parent, base, command, batter, routes, month
        )
    )
    np.testing.assert_allclose(baseline, 0.63)
    # Row 0 uses 60% command: fallback=.61, then deployed route weight=.30.
    assert np.isclose(candidate[0], 0.603)
    # Row 1 is the complement and uses 25% batter: fallback=.625.
    assert np.isclose(candidate[1], 0.6075)
    np.testing.assert_allclose(candidate[2:], baseline[2:])
    np.testing.assert_array_equal(selected, [True, True, False, False])
    np.testing.assert_array_equal(command_selected, [True, False, False, False])
