from __future__ import annotations

from src.archive.v226_recent2_fallback_replacement_audit import (
    locked_point_gate,
    restrictions,
    source_sanity,
)


def _result(gain: float, positive: float = 1.0, worst: float = 0.0) -> dict:
    return {
        "gain": gain,
        "positive_month_fraction": positive,
        "worst_month_gain": worst,
        "minimum_domain_gain": 0.0,
    }


def test_source_sanity_requires_positive_absolute_gain_on_both_sources() -> None:
    results = {"full_2022": _result(1.0), "late_2023": _result(0.1)}
    assert source_sanity(results)
    results["late_2023"] = _result(-0.1)
    assert not source_sanity(results)


def test_locked_point_gate_requires_material_incremental_gain() -> None:
    assert locked_point_gate(_result(0.1), _result(1.0, 0.625, -4.9))
    assert not locked_point_gate(_result(0.1), _result(0.99, 1.0, 0.0))
    assert not locked_point_gate(_result(0.1), _result(2.0, 0.50, 0.0))
    assert not locked_point_gate(_result(-0.1), _result(2.0, 1.0, 0.0))


def test_v226_restrictions_freeze_the_public1175_formula() -> None:
    audit = restrictions()
    assert audit["pressure_gate_frozen"]
    assert audit["parent_probability_threshold_frozen_at_050"]
    assert audit["fallback_weight_frozen_at_030"]
    assert audit["single_recent2_candidate_only"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
