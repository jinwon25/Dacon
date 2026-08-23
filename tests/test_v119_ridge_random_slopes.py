import numpy as np
import pandas as pd

from src.archive.v119_ridge_random_slopes import design_matrix, fit_state


def _rows(pitchers):
    n = len(pitchers)
    return pd.DataFrame(
        {
            "pitcher_id": pitchers,
            "batter_id": np.arange(n) + 10,
            "pitcher_team_id": np.ones(n, dtype=int),
            "batter_team_id": np.full(n, 2, dtype=int),
            "count_state": ["0-0"] * n,
            "pitcher_hand": np.ones(n, dtype=int),
            "batter_hand": np.full(n, 2, dtype=int),
            "balls_before": np.zeros(n),
            "strikes_before": np.zeros(n),
            "outs_before": np.zeros(n),
            "num_runners_on": np.zeros(n),
            "li": np.ones(n),
        }
    )


def test_unknown_entities_have_no_entity_columns():
    source = _rows([1, 2])
    state = fit_state(source)
    query = _rows([999])
    query["batter_id"] = 999
    matrix = design_matrix(query, state).toarray()
    pitcher_width = state.widths[0]
    batter_width = state.widths[1]
    assert np.all(matrix[:, :pitcher_width] == 0.0)
    assert np.all(matrix[:, pitcher_width : pitcher_width + batter_width] == 0.0)


def test_design_is_row_order_equivariant():
    source = _rows([1, 2, 1])
    state = fit_state(source)
    original = design_matrix(source, state).toarray()
    order = np.array([2, 0, 1])
    shuffled = design_matrix(source.iloc[order].reset_index(drop=True), state).toarray()
    np.testing.assert_allclose(shuffled, original[order])
