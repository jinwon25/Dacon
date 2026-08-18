import numpy as np
import pandas as pd

from src.v46_sparse_logit_screen import interaction_frame, matrix_pair


def test_interaction_frame_is_row_local_and_preserves_order():
    rows = pd.DataFrame(
        {
            "pitcher_id": [10, 20],
            "batter_id": [30, 40],
            "pitcher_team_id": [1, 2],
            "batter_team_id": [3, 4],
            "pitcher_hand": [1, 2],
            "batter_hand": [2, 1],
            "game_type": ["R", "F"],
            "domain3": ["R_CORE", "F"],
            "balls_before": [3, 0],
            "strikes_before": [1, 2],
            "base_state": ["1__", "___"],
            "inning": [8, 2],
            "score_diff_pitcher_team": [-2, 4],
            "li": [2.0, 0.5],
        }
    )
    output = interaction_frame(rows)
    assert output.index.tolist() == [0, 1]
    assert output["pitcher_count"].tolist() == ["10|3-1", "20|0-2"]
    assert output["pitcher_opponent_hand"].tolist() == ["10|2", "20|1"]
    assert output["domain_count"].tolist() == ["R_CORE|3-1", "F|0-2"]


def test_matrix_pair_fits_categories_and_scaling_on_source_only():
    fit_numeric = pd.DataFrame({"x": [0.0, 2.0], "constant": [1.0, 1.0]})
    audit_numeric = pd.DataFrame({"x": [100.0], "constant": [8.0]})
    fit_category = pd.DataFrame({"category": ["known", "known"]})
    audit_category = pd.DataFrame({"category": ["unseen"]})
    fit, audit, shape = matrix_pair(
        fit_numeric, audit_numeric, fit_category, audit_category
    )
    assert fit.shape[0] == 2
    assert audit.shape[0] == 1
    assert shape["numeric"] == 2
    # Numeric transforms are clipped; the unseen audit category is ignored.
    assert np.max(np.abs(audit.toarray()[:, :2])) <= 8.0
    assert np.allclose(audit.toarray()[:, 2:], 0.0)
