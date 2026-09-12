from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v177_forward_context_residual_eb import (
    add_context,
    apply_correction,
    fit_table,
    predict_table,
)


def _frame() -> pd.DataFrame:
    return add_context(
        pd.DataFrame(
            {
                "game_month": [8, 8, 9],
                "balls_before": [3, 0, 1],
                "strikes_before": [1, 2, 1],
                "num_runners_on": [1, 0, 0],
                "pitcher_hand": ["R", "L", "R"],
                "batter_hand": ["L", "L", "R"],
                "pitcher_id": [1, 2, 3],
                "pitcher_team_id": [10, 20, 30],
                "batter_team_id": [20, 30, 10],
            }
        )
    )


def test_fit_table_uses_only_masked_residuals_and_shrinks() -> None:
    frame = _frame()
    table = fit_table(
        frame,
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([0.4, 0.4, 0.4]),
        np.asarray([True, False, False]),
        ("game_month",),
        2.0,
    )
    prediction = predict_table(table, frame)
    assert np.allclose(prediction[:2], 0.03)  # capped after 0.6 / (1 + 2)
    assert prediction[2] == 0.0


def test_unseen_context_is_neutral_and_query_values_do_not_refit() -> None:
    frame = _frame()
    table = fit_table(
        frame,
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([0.5, 0.5, 0.5]),
        np.asarray([True, True, False]),
        ("game_month", "pitcher_id"),
        200.0,
    )
    query = frame.iloc[[2]].copy()
    assert predict_table(table, query)[0] == 0.0


def test_apply_correction_preserves_inactive_rows() -> None:
    parent = np.asarray([0.4, 0.5, 0.6])
    candidate = apply_correction(
        parent, np.asarray([0.1, -0.2, 0.3]), np.asarray([True, False, True]), 0.5
    )
    assert np.allclose(candidate, [0.45, 0.5, 0.75])
