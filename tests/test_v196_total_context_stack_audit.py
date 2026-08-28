from __future__ import annotations

from src.archive.v196_total_context_stack_audit import restrictions


def test_v196_restrictions_forbid_test_and_public_selection() -> None:
    contract = restrictions()
    assert contract["official_train_only"] is True
    assert contract["total_candidate_compared_to_exact_jy_parent"] is True
    assert contract["all_predeclared_v195_gates_in_reality_check"] is True
    assert contract["test_csv_read"] is False
    assert contract["test_aggregate_used"] is False
    assert contract["other_test_rows_required"] is False
    assert contract["public_score_used_for_selection"] is False
    assert contract["row_local_inference"] is True
