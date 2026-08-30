from __future__ import annotations

import numpy as np

from src.archive.v222_rcore_joint_h1_residual_audit import (
    apply_rcore_delta,
    restrictions,
    select_scale,
)


def test_apply_rcore_delta_changes_only_rcore() -> None:
    axis = {"domain3": np.array(["R_CORE", "F", "R_ANCHOR"])}
    result, active = apply_rcore_delta(
        np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2]), axis, 0.1
    )
    np.testing.assert_allclose(result, [0.52, 0.5, 0.5])
    assert active.tolist() == [True, False, False]


def test_select_scale_uses_sources_only() -> None:
    def item(gain: float):
        return {
            "gain": gain,
            "positive_month_fraction": 1.0,
            "worst_month_gain": 0.0,
            "minimum_domain_gain": 0.0,
        }
    results = {
        "0.02": {"full_2022": item(1.0), "late_2023": item(2.0)},
        "0.05": {"full_2022": item(1.5), "late_2023": item(1.5)},
    }
    assert select_scale(results) == 0.05


def test_v222_is_paired_and_test_free() -> None:
    audit = restrictions()
    assert audit["paired_three_seed_joint_h1_delta_only"]
    assert audit["all_rcore_scope_fixed_before_audit"]
    assert audit["public1175_fallback_formula_frozen"]
    assert not audit["test_csv_read"]
