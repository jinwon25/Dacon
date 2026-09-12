from __future__ import annotations

from src.archive.v229_latest_trackman_fallback_replacement_audit import restrictions


def test_v229_restrictions_freeze_public1175_contract() -> None:
    audit = restrictions()
    assert audit["public1175_evidence_parent_frozen"]
    assert audit["pressure_gate_frozen"]
    assert audit["parent_probability_threshold_frozen_at_050"]
    assert audit["fallback_weight_frozen_at_030"]
    assert audit["single_latest_trackman_candidate_only"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
