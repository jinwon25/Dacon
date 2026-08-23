from __future__ import annotations

import numpy as np
import pandas as pd

from src.v93_conditional_fm_benefit_gate import (
    apply_fm_gate,
    build_gate_features,
    fit_benefit_model,
    predict_gate,
    row_brier_benefit,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "domain3": ["R_CORE", "F", "R_CORE", "R_ANCHOR"],
            "control_success": [1, 0, 0, 1],
            "game_month": [4, 5, 6, 7],
            "pitcher_hand": ["R", "L", "R", "L"],
            "batter_hand": ["R", "R", "L", "L"],
            "top_bottom": ["top", "bottom", "top", "bottom"],
        }
    )


def test_gate_features_are_row_order_equivariant() -> None:
    frame = _frame()
    parent = np.array([0.4, 0.5, 0.6, 0.7])
    older = np.array([0.1, -0.1, 0.2, -0.2])
    recent = np.array([0.2, -0.2, -0.1, 0.1])
    original = build_gate_features(frame, parent, older, recent)
    order = np.array([2, 0, 3, 1])
    shuffled = build_gate_features(
        frame.iloc[order].reset_index(drop=True), parent[order], older[order], recent[order]
    )
    np.testing.assert_allclose(shuffled.to_numpy(), original.to_numpy()[order])


def test_fm_gate_changes_only_r_core_and_is_bounded() -> None:
    frame = _frame()
    parent = np.full(4, 0.5)
    output, active = apply_fm_gate(frame, parent, np.ones(4), np.array([1.0, 1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(active, [True, False, True, False])
    assert output[0] > 0.5
    np.testing.assert_allclose(output[1:], 0.5)
    assert np.all((output >= 0.001) & (output <= 0.999))


def test_row_benefit_is_zero_off_route() -> None:
    frame = _frame()
    benefit = row_brier_benefit(frame, np.full(4, 0.5), np.ones(4))
    assert benefit[1] == 0.0
    assert benefit[3] == 0.0
    assert benefit[0] > 0.0
    assert benefit[2] < 0.0


def test_small_ridge_gate_is_finite_and_bounded() -> None:
    frame = pd.concat([_frame()] * 10, ignore_index=True)
    parent = np.tile(np.array([0.4, 0.5, 0.6, 0.7]), 10)
    older = np.tile(np.array([0.1, -0.1, 0.2, -0.2]), 10)
    recent = np.tile(np.array([0.2, -0.2, -0.1, 0.1]), 10)
    features = build_gate_features(frame, parent, older, recent)
    benefit = np.linspace(-1.0, 1.0, len(frame))
    model = fit_benefit_model(features, benefit, np.ones(len(frame)), "ridge")
    for style in ("hard_positive", "soft_positive"):
        gate, expected = predict_gate(model, features, style)
        assert np.isfinite(expected).all()
        assert np.all((gate >= 0.0) & (gate <= 1.0))
