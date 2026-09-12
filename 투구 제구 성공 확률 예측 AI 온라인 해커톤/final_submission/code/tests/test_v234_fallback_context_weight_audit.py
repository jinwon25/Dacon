from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v234_fallback_context_weight_audit import (
    apply_context_weight,
    context_gates,
    restrictions,
)


def test_context_gates_are_row_local_and_weight_change_is_scoped() -> None:
    frame = pd.DataFrame({
        "balls_before": [3, 0], "strikes_before": [2, 1],
        "inning": [8, 2], "score_diff_pitcher_team": [0, 3],
        "num_runners_on": [2, 0], "li": [2.0, 0.2],
        "pitcher_hand": ["R", "L"], "batter_hand": ["R", "R"],
    })
    parent = np.array([0.55, 0.51])
    gates = context_gates(frame, parent)
    assert gates["three_ball"].tolist() == [True, False]
    assert gates["late_inning"].tolist() == [True, False]
    current = np.array([0.565, 0.515])
    xgb = np.array([0.60, 0.55])
    output, changed = apply_context_weight(
        current, parent, xgb, np.array([True, True]),
        gates["three_ball"], 0.50,
    )
    assert changed.tolist() == [True, False]
    np.testing.assert_allclose(output, [0.575, 0.515])


def test_v234_restrictions_keep_one_row_local_rule() -> None:
    audit = restrictions()
    assert audit["fallback_xgb_model_frozen"]
    assert audit["public1175_active_support_frozen"]
    assert audit["one_context_rule_only"]
    assert audit["row_local_baseball_contexts_only"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
