import numpy as np
import pytest

from experiments.group1_recency_overlay import compose_recency_overlay, select_alpha


def test_compose_recency_overlay_adds_only_model_difference_and_clips() -> None:
    incumbent = np.array([100.0, 900.0])
    equal = np.array([200.0, 800.0])
    recent = np.array([0.0, 1_200.0])
    result = compose_recency_overlay(
        incumbent, equal, recent, alpha=0.5, capacity=1_000.0,
    )
    np.testing.assert_allclose(result, [0.0, 1_000.0])


def test_compose_recency_overlay_rejects_negative_strength() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compose_recency_overlay(
            np.ones(2), np.ones(2), np.ones(2), alpha=-0.1,
        )


def test_select_alpha_prefers_zero_when_recency_factor_is_harmful() -> None:
    truth = np.array([10_000.0, 12_000.0, 14_000.0])
    incumbent = truth.copy()
    equal = truth.copy()
    recent = truth + 4_000.0
    selected, metrics = select_alpha(
        truth,
        incumbent,
        equal,
        recent,
        np.ones(3, dtype=bool),
        (0.0, 0.5, 1.0),
    )
    assert selected == 0.0
    assert metrics["0.0"]["score"] > metrics["0.5"]["score"]
