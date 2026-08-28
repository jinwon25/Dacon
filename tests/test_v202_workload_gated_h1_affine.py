from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v202_workload_gated_h1_affine import (
    apply_gated_affine,
    h1_affine_top_direction,
    restrictions,
    role_gates,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_runners_on": [0, 1],
            "li": [1.0, 1.0],
            "asof_pitcher_prev1_game_success_rate": [0.350877, 0.5],
            "asof_pitcher_prev1_game_middle_rate": [0.157895, 0.25],
            "asof_pitcher_prev5_game_success_rate": [0.55, 0.5],
            "asof_pitcher_prev5_game_middle_rate": [0.20, 0.25],
        }
    )


def test_role_gates_use_inferred_workload() -> None:
    gates = role_gates(_frame())
    assert gates["starter_prev1"].tolist() == [True, False]
    assert gates["short_prev1"].tolist() == [False, True]


def test_h1_top_direction_respects_champion_route_weight() -> None:
    direction = h1_affine_top_direction(np.array([0.60, 0.60]), _frame())
    assert direction[1] > direction[0] > 0.0


def test_apply_gated_affine_only_changes_exact_rcore_gate() -> None:
    candidate, active = apply_gated_affine(
        np.array([0.5, 0.5, 0.5]),
        np.array([0.01, 0.01, 0.01]),
        np.array([True, True, False]),
        np.array([True, False, True]),
        np.array(["R_CORE", "R_CORE", "R_CORE"]),
        -1.0,
    )
    assert active.tolist() == [True, False, False]
    assert np.allclose(candidate, [0.495, 0.5, 0.5])


def test_v202_restrictions_are_strict_and_row_local() -> None:
    audit = restrictions()
    assert audit["fixed_baseball_role_gates"]
    assert audit["fixed_affine_and_half_dose"]
    assert audit["source_selection_before_locked_2024"]
    assert audit["row_local_inference"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
