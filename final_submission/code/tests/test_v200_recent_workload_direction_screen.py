from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v200_recent_workload_direction_screen import (
    apply_direction,
    restrictions,
    workload_directions,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_pitcher_success_rate": [0.60, 0.55],
            "asof_pitcher_middle_rate": [0.20, 0.25],
            "asof_pitcher_prev1_game_success_rate": [0.50, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.55, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.58, np.nan],
            "asof_pitcher_prev1_game_middle_rate": [0.25, np.nan],
            "asof_pitcher_prev3_game_middle_rate": [0.20, np.nan],
            "asof_pitcher_prev5_game_middle_rate": [0.22, np.nan],
        }
    )


def test_workload_directions_are_row_local_and_missing_neutral() -> None:
    directions = workload_directions(_frame())
    assert set(directions) == {
        "success_prev1_shrink",
        "success_prev3_shrink",
        "success_prev5_shrink",
        "success_shrink_consensus",
        "middle_shrink_consensus",
        "success_denominator_reweight",
        "middle_denominator_reweight",
    }
    assert directions["success_prev1_shrink"][0] > 0.0
    assert all(values[1] == 0.0 for values in directions.values())


def test_apply_direction_only_changes_exact_rcore_nonzero_rows() -> None:
    parent = np.array([0.5, 0.5, 0.5, 0.5])
    direction = np.array([0.1, 0.1, 0.1, 0.0])
    candidate, active = apply_direction(
        parent,
        direction,
        np.array([True, False, True, True]),
        np.array(["R_CORE", "R_CORE", "F", "R_CORE"]),
        0.01,
    )
    assert active.tolist() == [True, False, False, False]
    assert np.allclose(candidate, [0.501, 0.5, 0.5, 0.5])


def test_v200_restrictions_prevent_test_and_public_selection() -> None:
    audit = restrictions()
    assert audit["official_train_only"]
    assert audit["fixed_target_free_direction_library"]
    assert audit["source_selection_before_locked_2024"]
    assert audit["row_local_inference"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
