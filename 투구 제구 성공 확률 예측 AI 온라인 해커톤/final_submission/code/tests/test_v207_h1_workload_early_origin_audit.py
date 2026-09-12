from __future__ import annotations

from src.archive.v207_h1_workload_early_origin_audit import (
    origin_gate,
    restrictions,
)


def test_origin_gate_requires_sign_months_and_tail() -> None:
    assert origin_gate(
        {"gain": 1.0, "positive_month_fraction": 0.5, "worst_month_gain": -4.9}
    )
    assert not origin_gate(
        {"gain": 1.0, "positive_month_fraction": 0.4, "worst_month_gain": -4.9}
    )
    assert not origin_gate(
        {"gain": 1.0, "positive_month_fraction": 0.6, "worst_month_gain": -5.0}
    )


def test_v207_uses_early_forward_origins_without_test_batch() -> None:
    audit = restrictions()
    assert audit["strictly_prior_season_fits"]
    assert audit["fixed_primary_scale"]
    assert audit["early_origins_not_used_by_v203_selection"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
