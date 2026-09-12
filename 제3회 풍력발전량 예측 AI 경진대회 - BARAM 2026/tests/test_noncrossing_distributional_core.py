from __future__ import annotations

import numpy as np
import torch

from experiments.noncrossing_distributional_core import (
    NonCrossingQuantileMLP,
    closest_utility_maximizer,
    distribution_point_forecasts,
    interpolate_distribution_action,
    pinball_loss,
)


def test_simplex_head_is_bounded_and_noncrossing() -> None:
    torch.manual_seed(7)
    model = NonCrossingQuantileMLP(5, hidden=16, dropout=0.0)
    values = model(torch.randn(12, 5)).detach().numpy()
    assert values.shape == (12, 10)
    assert np.all(values > 0.0)
    assert np.all(values < 1.0)
    assert np.all(np.diff(values, axis=1) >= 0.0)


def test_pinball_loss_is_finite() -> None:
    prediction = torch.linspace(0.1, 0.9, 10).repeat(4, 1)
    truth = torch.tensor([0.2, 0.4, 0.6, 0.8])
    assert torch.isfinite(pinball_loss(prediction, truth))


def test_distribution_action_is_bounded() -> None:
    quantiles = np.tile(np.arange(0.05, 1.0, 0.10), (20, 1))
    median, action, diagnostics = distribution_point_forecasts(
        quantiles,
        capacity=21_600.0,
        mean_generation=10_000.0,
    )
    assert median.shape == action.shape == (20,)
    assert np.all((action >= 0.0) & (action <= 21_600.0))
    assert diagnostics["maximum_action_from_median_ratio"] <= 0.0400001


def test_distribution_action_interpolation_keeps_its_own_median_anchor() -> None:
    median = {
        "kpx_group_1": np.asarray([1_000.0, 2_000.0]),
        "kpx_group_2": np.asarray([3_000.0, 4_000.0]),
    }
    action = {
        "kpx_group_1": np.asarray([1_400.0, 1_600.0]),
        "kpx_group_2": np.asarray([2_600.0, 4_400.0]),
    }
    candidate = interpolate_distribution_action(median, action, 0.25)
    assert np.allclose(candidate["kpx_group_1"], [1_100.0, 1_900.0])
    assert np.allclose(candidate["kpx_group_2"], [2_900.0, 4_100.0])


def test_utility_plateau_prefers_action_nearest_reference() -> None:
    utility = np.asarray(
        [
            [0.4, 0.4, 0.4],
            [0.5, 0.6, 0.6],
        ]
    )
    candidates = np.asarray(
        [
            [80.0, 100.0, 120.0],
            [80.0, 100.0, 120.0],
        ]
    )
    selected = closest_utility_maximizer(
        utility,
        candidates,
        np.asarray([100.0, 115.0]),
    )
    assert selected.tolist() == [1, 2]
