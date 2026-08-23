from __future__ import annotations

import numpy as np
import pandas as pd

from src.v62_group_balanced_extra_trees import (
    blend_candidate,
    group_weights,
    matrix_pair,
    prepare_rows,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e", "f"],
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "game_month": [3, 4, 5, 3, 4, 5],
            "game_dayofweek": [1, 2, 3, 1, 2, 3],
            "inning": [1, 2, 3, 4, 5, 6],
            "top_bottom": ["T", "B", "T", "B", "T", "B"],
            "game_type": ["R", "R", "F", "R", "R", "F"],
            "balls_before": [0, 3, 1, 2, 0, 3],
            "strikes_before": [0, 1, 2, 1, 2, 0],
            "outs_before": [0, 1, 2, 0, 1, 2],
            "base_state": ["___", "1__", "_2_", "__3", "12_", "123"],
            "score_diff_pitcher_team": [0, 1, -1, 2, -2, 0],
            "pitcher_id": [1, 2, 3, 1, 2, 3],
            "batter_id": [11, 12, 13, 14, 15, 16],
            "pitcher_hand": [1, 2, 1, 2, 1, 2],
            "batter_hand": [2, 1, 2, 1, 1, 2],
            "pitcher_team_id": [1, 2, 13, 1, 2, 13],
            "batter_team_id": [3, 13, 4, 3, 13, 4],
            "asof_pitcher_n": [10, 20, 30, 40, 50, 60],
            "asof_pitcher_success_rate": [0.5] * 6,
            "asof_pitcher_prev1_game_success_rate": [0.4] * 6,
            "asof_pitcher_prev3_game_success_rate": [0.5] * 6,
            "asof_pitcher_prev5_game_success_rate": [0.6] * 6,
            "asof_pitcher_strike_rate": [0.3] * 6,
            "asof_pitcher_ball_rate": [0.2] * 6,
            "asof_pitcher_middle_rate": [0.2] * 6,
            "asof_pitcher_reverse_rate": [0.1] * 6,
            "asof_batter_n": [20] * 6,
            "asof_pitcher_pitchmix_n": [10] * 6,
            "asof_pitcher_fastball_rate": [0.5] * 6,
            "asof_pitcher_breaking_rate": [0.3] * 6,
            "asof_pitcher_offspeed_rate": [0.2] * 6,
            "control_success": [1, 0, 1, 0, 1, 0],
        }
    )


def test_group_equal_weights_have_equal_total_group_mass() -> None:
    frame = prepare_rows(_rows())
    weight = group_weights(frame, "season_domain_equal")
    keys = frame["season"].astype(str) + ":" + frame["domain3"].astype(str)
    masses = [weight[keys.eq(key)].sum() for key in sorted(keys.unique())]
    assert np.allclose(masses, masses[0])
    assert np.isclose(weight.mean(), 1.0)


def test_matrix_pair_does_not_learn_query_only_category() -> None:
    source = pd.concat([_rows()] * 5, ignore_index=True)
    query = _rows().iloc[[0]].copy()
    query["pitcher_team_id"] = 999
    source = prepare_rows(source)
    query = prepare_rows(query)
    left, right, schema = matrix_pair(source, query)
    assert left.shape[1] == right.shape[1] == schema["total_count"]
    assert not any(name == "pitcher_team_id=999" for name in schema["categorical_features"])
    assert np.isfinite(right).all()


def test_row_features_are_partition_invariant() -> None:
    source = prepare_rows(pd.concat([_rows()] * 5, ignore_index=True))
    query = prepare_rows(_rows())
    _, full, _ = matrix_pair(source, query)
    partitions = []
    for index in range(len(query)):
        _, single, _ = matrix_pair(source, query.iloc[[index]].copy())
        partitions.append(single)
    assert np.array_equal(full, np.vstack(partitions))


def test_blend_changes_only_requested_domain() -> None:
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, 0.0],
            "game_month": [3, 3, 3],
            "domain3": ["R_CORE", "R_ANCHOR", "F"],
        }
    )
    parent = np.array([0.5, 0.5, 0.5])
    challenger = np.array([0.7, 0.2, 0.9])
    candidate, active = blend_candidate(frame, parent, challenger, "R_CORE", 0.1)
    assert np.array_equal(active, np.array([True, False, False]))
    assert np.allclose(candidate, [0.52, 0.5, 0.5])
