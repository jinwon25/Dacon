import pytest
import torch

from experiments.spatiotemporal_ficr_loss_tune import (
    choose_development_policy,
    day_bootstrap_delta,
    official_boundary_loss,
)


def test_official_boundary_loss_is_finite_and_differentiable() -> None:
    prediction = torch.tensor([[[0.25, 0.30, 0.40]]], requires_grad=True)
    target = torch.tensor([[[0.20, 0.37, float("nan")]]])
    loss = official_boundary_loss(prediction, target, 0.02, 0.004)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_official_boundary_loss_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        official_boundary_loss(torch.zeros(1, 1, 3), torch.ones(1, 1, 3), 0.02, 0.0)


def test_policy_selector_requires_seed_and_component_transfer() -> None:
    good = {
        "variant": "good",
        "alpha": 0.2,
        "q1": {"score": 0.02, "one_minus_nmae": 0.01, "ficr": 0.03},
        "q2": {"score": 0.03, "one_minus_nmae": 0.01, "ficr": 0.05},
        "seed_score_deltas": {"q1": [0.01, 0.02], "q2": [0.02, 0.03]},
    }
    brittle = {
        "variant": "brittle",
        "alpha": 0.3,
        "q1": {"score": 0.04, "one_minus_nmae": 0.01, "ficr": 0.07},
        "q2": {"score": 0.04, "one_minus_nmae": 0.01, "ficr": 0.07},
        "seed_score_deltas": {"q1": [-0.01, 0.08], "q2": [0.01, 0.07]},
    }
    assert choose_development_policy([good, brittle])["variant"] == "good"


def test_day_bootstrap_preserves_a_strictly_better_candidate() -> None:
    import numpy as np
    import pandas as pd

    timestamps = pd.date_range("2024-07-01", periods=48, freq="h")
    truth = np.full(48, 5_000.0)
    base = np.full(48, 7_000.0)
    candidate = np.full(48, 5_100.0)
    result = day_bootstrap_delta(
        truth, base, candidate, timestamps, np.ones(48, dtype=bool), n_bootstrap=20
    )
    assert result["positive_fraction"] == 1.0
    assert result["q05"] > 0.0
