from __future__ import annotations

from src.archive.v235_hoo_state_catboost_public1175_screen import restrictions


def test_v235_is_research_only_until_local_reimplementation() -> None:
    audit = restrictions()
    assert audit["external_predictions_research_evidence_only"]
    assert not audit["external_code_or_weights_in_package"]
    assert audit["local_reimplementation_required_before_packaging"]
    assert audit["source_only_family_route_weight_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
