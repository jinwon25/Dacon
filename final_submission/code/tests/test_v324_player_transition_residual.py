import numpy as np
import pandas as pd

from src.archive.v324_player_transition_residual import (
    entity_transition_state,
    transition_features,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022, 2022],
            "pitcher_id": [1, 2, 1, 2, 4],
            "batter_id": [11, 12, 11, 12, 14],
            "pitcher_team_id": [10, 20, 10, 20, 40],
            "batter_team_id": [110, 120, 110, 120, 140],
            "balls_before": [0, 1, 2, 3, 0],
            "strikes_before": [0, 1, 2, 0, 1],
        }
    )


def test_entity_state_distinguishes_new_return_same_and_switch() -> None:
    train = _train()
    query = pd.DataFrame(
        {
            "pitcher_id": [1, 2, 3, 4],
            "pitcher_team_id": [10, 99, 30, 40],
        }
    )
    state = entity_transition_state(
        train, query, 2023, "pitcher_id", "pitcher_team_id"
    )
    assert state["status"].tolist() == ["SAME", "SWITCH", "NEW", "SAME"]


def test_future_rows_do_not_change_prior_transition_features() -> None:
    train = _train()
    query = pd.DataFrame(
        {
            "pitcher_id": [1], "batter_id": [11],
            "pitcher_team_id": [10], "batter_team_id": [110],
            "balls_before": [3], "strikes_before": [2],
        }
    )
    before = transition_features(train, query, 2023)
    future = train.iloc[[0]].copy()
    future["season"] = 2024
    future["pitcher_team_id"] = 999
    after = transition_features(pd.concat([train, future]), query, 2023)
    pd.testing.assert_frame_equal(before, after)


def test_query_rows_are_mapped_independently() -> None:
    train = _train()
    query = pd.DataFrame(
        {
            "pitcher_id": [1, 2], "batter_id": [11, 12],
            "pitcher_team_id": [10, 99], "batter_team_id": [110, 999],
            "balls_before": [0, 3], "strikes_before": [0, 2],
        }
    )
    full = transition_features(train, query, 2023).reset_index(drop=True)
    singles = pd.concat(
        [transition_features(train, query.iloc[[i]].reset_index(drop=True), 2023) for i in range(2)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(full, singles)
    assert np.array_equal(full["pitcher_status"], np.asarray(["SAME", "SWITCH"]))
