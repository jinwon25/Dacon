from __future__ import annotations

from src.archive.v230_latest_trackman_complement_scope_audit import restrictions


def test_v230_restrictions_keep_latest_model_disjoint() -> None:
    audit = restrictions()
    assert audit["current_public1175_active_model_frozen"]
    assert audit["latest_trackman_model_used_only_on_disjoint_complements"]
    assert audit["candidate_scopes_preregistered_from_v224"]
    assert audit["source_only_scope_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
