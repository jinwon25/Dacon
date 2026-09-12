from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v206_augmented_h1_weight_scope_audit import (
    restrictions,
    scope_mask,
    scoped_replacement,
)


def _axis() -> dict[str, np.ndarray]:
    return {"domain3": np.array(["R_CORE", "R_CORE", "F"])}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"num_runners_on": [0, 1, 0], "li": [1.0, 1.0, 2.0]}
    )


def test_scope_masks_split_rcore_without_overlap() -> None:
    all_rows = scope_mask(_axis(), _frame(), "all_rcore")
    jy = scope_mask(_axis(), _frame(), "pressure_gate")
    non_jy = scope_mask(_axis(), _frame(), "non_pressure_gate")
    assert all_rows.tolist() == [True, True, False]
    assert jy.tolist() == [False, True, False]
    assert non_jy.tolist() == [True, False, False]
    assert np.array_equal(jy | non_jy, all_rows)


def test_unknown_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        scope_mask(_axis(), _frame(), "month_tuned")


def test_scoped_replacement_preserves_inactive_rows() -> None:
    output = scoped_replacement(
        np.array([0.2, 0.4, 0.6]),
        np.array([0.3, 0.5, 0.7]),
        np.array([False, True, False]),
    )
    assert np.allclose(output, [0.2, 0.5, 0.6])


def test_v206_restrictions_forbid_test_batch_and_public_selection() -> None:
    audit = restrictions()
    assert audit["fixed_weight_lifts_and_baseball_scopes"]
    assert audit["source_selection_before_locked_2024"]
    assert audit["three_seed_confirmation_required"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
