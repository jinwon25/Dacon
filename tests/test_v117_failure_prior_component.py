from __future__ import annotations

import numpy as np
import pandas as pd

from src.v117_failure_prior_component import (
    build_failure_bank,
    predict_failure_components,
)
from src.v117_v116_probe_wrapper import apply_v116


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "pitcher_id": [1, 1, 1, 1],
            "batter_hand": [1, 1, 2, 2],
            "count_state": [0, 0, 1, 1],
            "pitch_type_fine": ["Fastball", "Slider", "Fastball", "Slider"],
            "failure__reverse": [0.0, 1.0, 0.0, 1.0],
            "failure__middle": [1.0, 0.0, 1.0, 0.0],
            "failure__wayoff": [0.0, 0.0, 1.0, 0.0],
        }
    )


def test_bank_predictions_are_row_order_independent() -> None:
    bank = build_failure_bank(
        _history(), ["Fastball", "Slider"], outcome_shrink=1.0,
        selection_shrink=1.0,
    )
    rows = _history().iloc[[0, 2]][["pitcher_id", "batter_hand", "count_state"]]
    full = predict_failure_components(rows, bank)
    reversed_prediction = predict_failure_components(rows.iloc[::-1], bank)[::-1]
    np.testing.assert_allclose(full, reversed_prediction, atol=0.0)


def test_apply_v116_protects_non_core_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "F"],
            "pitcher_team_id": [1, 13, 1],
            "batter_team_id": [2, 2, 2],
        }
    )
    parent = np.array([0.4, 0.5, 0.6])
    actual = apply_v116(parent, frame, np.array([0.1, 0.1, 0.1]))
    np.testing.assert_allclose(actual, np.array([0.5, 0.5, 0.6]))
