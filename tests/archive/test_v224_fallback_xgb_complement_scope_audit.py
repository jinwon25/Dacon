from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v224_fallback_xgb_complement_scope_audit import (
    apply_additional_fallback,
    complement_scopes,
    restrictions,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "game_type": ["R"] * 4 + ["F"],
        "pitcher_team_id": [1] * 5,
        "batter_team_id": [2] * 5,
        "num_runners_on": [1, 1, 0, 0, 1],
        "li": [1.0] * 5,
    })


def test_complement_scopes_are_disjoint_from_deployed() -> None:
    parent = np.array([0.6, 0.4, 0.6, 0.4, 0.6])
    scopes = complement_scopes(parent, _frame())
    deployed = np.array([True, False, False, False, False])
    assert all(not np.any(mask & deployed) for mask in scopes.values())
    assert scopes["pressure_low50"].tolist() == [False, True, False, False, False]
    assert scopes["nonpressure_all"].tolist() == [False, False, True, True, False]


def test_additional_fallback_reuses_thirty_percent() -> None:
    result = apply_additional_fallback(
        np.array([0.4, 0.6]), np.array([0.6, 0.2]), np.array([True, False])
    )
    np.testing.assert_allclose(result, [0.46, 0.6])


def test_v224_does_not_retune_public1175_weight() -> None:
    audit = restrictions()
    assert audit["fallback_weight_frozen_at_public1175"]
    assert audit["candidate_scopes_disjoint_from_deployed"]
    assert not audit["test_csv_read"]
