from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.kma_group3_regime_power_curve import (
    AdjustmentPolicy,
    apply_regime_adjustment,
    fit_regime_power,
    regime_codes,
    select_adjustment_policy,
)
from experiments.kma_um_power_curve_gate import CAPACITY


def _context(rows: int) -> pd.DataFrame:
    angle = np.linspace(-np.pi, np.pi, rows, endpoint=False)
    return pd.DataFrame(
        {
            "kma_um_ctx_u10_r0": np.cos(angle),
            "kma_um_ctx_v10_r0": np.sin(angle),
            "kma_um_ctx_speed10_r0": np.linspace(2.0, 15.0, rows),
            "kma_um_ctx_shear10_850_r0": np.linspace(1.0, 8.0, rows),
            "kma_um_ctx_cos10_850_r0": np.linspace(0.85, 1.0, rows),
            "kma_um_ctx_run_change10": np.linspace(-2.0, 2.0, rows),
        }
    )


def test_regime_edges_are_fit_only_on_training_rows() -> None:
    context = _context(12)
    train = np.zeros(12, dtype=bool)
    train[:6] = True
    _, predict_codes, metadata = regime_codes(
        "shear3", context, train, context
    )
    expected = np.quantile(
        context.loc[:5, "kma_um_ctx_shear10_850_r0"],
        (1 / 3, 2 / 3),
    )
    assert metadata["edges"] == pytest.approx(expected)
    assert predict_codes[-1] == 2


def test_joint_direction_shear_regime_has_at_most_eight_codes() -> None:
    context = _context(1_200)
    train = np.ones(len(context), dtype=bool)
    train_codes, predict_codes, _ = regime_codes(
        "direction4_shear2", context, train, context
    )
    assert set(np.unique(train_codes)) <= set(range(8))
    assert np.array_equal(train_codes, predict_codes)


def test_regime_power_is_finite_and_shrunk_toward_global() -> None:
    rows = 1_200
    context = _context(rows)
    wind = context["kma_um_ctx_speed10_r0"].to_numpy(dtype=float)
    proxy = np.clip(1_000.0 * wind**1.3, 0.0, CAPACITY)
    proxy[rows // 2 :] += 500.0
    prediction, global_prediction, metadata = fit_regime_power(
        wind,
        proxy,
        context,
        np.ones(rows, dtype=bool),
        wind,
        context,
        "direction4",
        minimum_regime_rows=100,
        shrinkage_rows=400.0,
    )
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction - global_prediction)) > 0.0
    assert any(row["fitted"] for row in metadata["regimes"].values())
    assert all(
        0.0 <= row["local_weight"] < 1.0
        for row in metadata["regimes"].values()
    )


def test_regime_adjustment_preserves_caps_and_untouched_rows() -> None:
    public_reference = np.full(5, 8_000.0)
    incumbent = np.asarray([8_100.0, 8_100.0, 8_100.0, 8_100.0, 8_100.0])
    global_direct = np.full(5, 9_000.0)
    regime_direct = np.asarray([10_000.0, 7_000.0, 10_000.0, 10_000.0, 10_000.0])
    base_gate = np.asarray([True, True, False, True, True])
    policy = AdjustmentPolicy(
        "direction4",
        "up",
        1.0,
        0.10,
        0.80,
        None,
        1.0,
    )
    candidate, gate = apply_regime_adjustment(
        public_reference,
        incumbent,
        global_direct,
        regime_direct,
        base_gate,
        policy,
        kma_alpha=0.2,
        maximum_total_movement_ratio=0.05,
        maximum_incremental_movement_ratio=0.01,
    )
    assert gate.tolist() == [True, False, False, True, True]
    assert candidate[1] == incumbent[1]
    assert candidate[2] == incumbent[2]
    assert np.max(np.abs(candidate - incumbent)) <= 0.01 * CAPACITY + 1e-9
    assert np.max(np.abs(candidate - public_reference)) <= 0.05 * CAPACITY + 1e-9


def test_policy_selection_enforces_predeclared_changed_ratio() -> None:
    rows = 120
    incumbent = np.full(rows, 8_000.0)
    global_direct = np.full(rows, 8_000.0)
    regime_direct = 8_000.0 + np.linspace(0.0, 2_000.0, rows)
    truth = np.full(rows, 10_000.0)
    selected = select_adjustment_policy(
        truth,
        incumbent,
        incumbent,
        global_direct,
        regime_direct,
        np.ones(rows, dtype=bool),
        np.ones(rows, dtype=bool),
        "direction4",
        kma_alpha=1.0,
        maximum_total_movement_ratio=0.05,
        maximum_incremental_movement_ratio=0.02,
        minimum_changed_rows=1,
        maximum_changed_ratio=0.10,
    )
    assert selected is not None
    assert selected["selected"]["changed_rows"] <= 12
