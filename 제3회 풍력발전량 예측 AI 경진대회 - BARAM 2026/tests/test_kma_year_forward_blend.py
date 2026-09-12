from __future__ import annotations

import numpy as np

from experiments.kma_year_forward_blend import (
    CAPACITY,
    apply_sparse_expert_blend,
    fit_mixed_direct_power,
    select_sparse_expert_blend,
)


def test_mixed_direct_power_stays_finite_and_bounded() -> None:
    history_wind = np.linspace(1.0, 15.0, 1_200)
    history_label = np.linspace(0.0, CAPACITY, 1_200)
    recent_wind = np.linspace(1.0, 15.0, 1_500)
    recent_label = np.linspace(0.0, CAPACITY, 1_500)
    train = np.zeros(1_500, dtype=bool)
    train[:1_100] = True
    available = np.ones(1_500, dtype=bool)
    prediction = fit_mixed_direct_power(
        history_wind,
        history_label,
        recent_wind,
        recent_label,
        train,
        available,
        np.zeros(1_500),
        history_weight=0.5,
    )
    assert np.isfinite(prediction).all()
    assert prediction.min() >= 0.0
    assert prediction.max() <= CAPACITY


def test_sparse_selector_respects_changed_ratio() -> None:
    rows = 400
    recent = np.full(rows, 8_000.0)
    prior = recent + np.linspace(1.0, 4_000.0, rows)
    truth = recent.copy()
    truth[-100:] = prior[-100:]
    available = np.ones(rows, dtype=bool)
    selection = np.ones(rows, dtype=bool)
    result = select_sparse_expert_blend(
        truth,
        recent,
        prior,
        available,
        selection,
        maximum_changed_ratio=0.25,
    )
    assert result is not None
    selected = result["selected"]
    assert selected["changed_ratio"] <= 0.25
    candidate, gate = apply_sparse_expert_blend(
        recent,
        prior,
        available,
        selected["policy"],
        selected["alpha"],
    )
    assert int(gate.sum()) == selected["changed_rows"]
    assert np.any(candidate != recent)
