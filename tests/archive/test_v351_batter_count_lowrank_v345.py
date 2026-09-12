from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v351_batter_count_lowrank_v345 import prepare, route_masks


def test_prepare_context_is_count_by_pitcher_hand() -> None:
    frame = pd.DataFrame({
        "balls_before": [0, 3],
        "strikes_before": [0, 2],
        "pitcher_hand": [1, 2],
    })
    assert prepare(frame)["context_position"].tolist() == [0, 23]


def test_routes_partition_known_rows() -> None:
    frame = pd.DataFrame({
        "game_type": ["F", "R", "R"],
        "pitcher_team_id": [1, 13, 1],
        "batter_team_id": [2, 2, 2],
    })
    masks = route_masks(frame)
    assert np.all(masks["F"].astype(int) + masks["R_ANCHOR"] + masks["R_CORE"] == 1)
