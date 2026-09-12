from __future__ import annotations

import numpy as np

from src.archive.v227_fallback_forward_calibration_audit import (
    calibrated_fallback,
    fit_calibrator,
    restrictions,
)


def test_fit_affine_recovers_xgb_leg_mapping() -> None:
    parent = np.array([0.50, 0.52, 0.54, 0.56])
    xgb = np.array([0.40, 0.50, 0.60, 0.70])
    target = 0.70 * parent + 0.30 * (0.10 + 0.80 * xgb)
    active = np.ones(4, dtype=bool)
    a, b = fit_calibrator(parent, xgb, target, active, "affine")
    np.testing.assert_allclose([a, b], [0.10, 0.80], atol=1e-12)


def test_offset_and_dose_preserve_inactive_rows() -> None:
    parent = np.array([0.55, 0.45])
    xgb = np.array([0.50, 0.60])
    pressure = np.array([True, True])
    output, active = calibrated_fallback(parent, xgb, pressure, (0.10, 1.0), 0.5)
    assert active.tolist() == [True, False]
    np.testing.assert_allclose(output, [0.70 * 0.55 + 0.30 * 0.55, 0.45])


def test_v227_restrictions_freeze_public1175_contract() -> None:
    audit = restrictions()
    assert audit["fallback_xgb_model_frozen"]
    assert audit["public1175_gate_frozen"]
    assert audit["public1175_weight_frozen_at_030"]
    assert audit["chronological_source_fits"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
