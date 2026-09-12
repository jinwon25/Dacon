import numpy as np
import pytest

from src.archive.v261_regular_calendar_expert_gate import apply_calendar_gate


def test_calendar_gate_preserves_march_and_october_parent() -> None:
    baseline = np.array([0.1, 0.2, 0.3, 0.4])
    expert = np.array([0.9, 0.8, 0.7, 0.6])
    active = np.ones(4, dtype=bool)
    candidate, selected = apply_calendar_gate(
        baseline, expert, active, np.array([3, 4, 9, 10])
    )
    np.testing.assert_allclose(candidate, [0.1, 0.8, 0.7, 0.4])
    assert selected.tolist() == [False, True, True, False]


def test_calendar_gate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        apply_calendar_gate(
            np.zeros(2), np.zeros(2), np.ones(2, dtype=bool), np.zeros(3)
        )
