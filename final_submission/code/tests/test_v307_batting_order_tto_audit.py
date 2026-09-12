import numpy as np
import pandas as pd

from src.archive.v307_batting_order_tto_audit import (
    _snap_to_slot,
    attach_batting_state,
)


def test_attach_batting_state_counts_pa_by_offensive_team() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"TRAIN_{index:07d}" for index in range(1, 9)],
            "season": [2021] * 8,
            "game_type": ["R"] * 8,
            "inning": [1, 1, 1, 1, 1, 1, 2, 2],
            "top_bottom": ["T", "T", "T", "T", "B", "B", "T", "T"],
            "balls_before": [0, 1, 0, 0, 0, 1, 0, 1],
            "strikes_before": [0, 0, 0, 0, 0, 0, 0, 0],
            "outs_before": [0] * 8,
            "run_top_before": [0] * 8,
            "run_bot_before": [0] * 8,
            "num_runners_on": [0] * 8,
            "runner_on_1b": [0] * 8,
            "runner_on_2b": [0] * 8,
            "runner_on_3b": [0] * 8,
            "batter_id": [10, 10, 11, 12, 20, 20, 13, 13],
            "batter_team_id": [1, 1, 1, 1, 2, 2, 1, 1],
        }
    )
    rows = attach_batting_state(frame)
    assert rows["_pa_index"].tolist() == [1, 1, 2, 3, 1, 1, 4, 4]
    assert rows["_lineup_slot"].tolist() == [1, 1, 2, 3, 1, 1, 4, 4]
    assert rows["_true_tto"].tolist() == [1] * 8


def test_snap_to_slot_selects_nearest_legal_cycle() -> None:
    expected = np.asarray([12.0, 20.0, 7.0, np.nan])
    slot = np.asarray([3.0, 1.0, np.nan, 5.0])
    snapped = _snap_to_slot(expected, slot)
    assert np.allclose(snapped[:2], [12.0, 19.0])
    assert snapped[2] == 7.0
    assert np.isnan(snapped[3])
