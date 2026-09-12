import numpy as np
import pandas as pd

from src.archive.hierarchical import HierarchicalBackoff


def _frame():
    return pd.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2],
            "batter_id": [10, 11, 10, 12],
            "pitcher_hand": [1, 1, 2, 2],
            "batter_hand": [2, 1, 2, 1],
            "pitcher_team_id": [3, 3, 4, 4],
            "game_type": ["R", "R", "R", "F"],
            "balls_before": [0, 1, 0, 3],
            "strikes_before": [0, 1, 2, 2],
            "base_state": ["___", "1__", "_2_", "123"],
            "inning": [1, 2, 7, 9],
            "li": [0.5, 1.0, 2.0, 4.0],
            "control_success": [1, 0, 1, 0],
        }
    )


def test_unknown_entities_back_off_without_nan():
    encoder = HierarchicalBackoff(alpha_leaf=10, alpha_parent=20).fit(_frame())
    unknown = _frame().iloc[[0]].drop(columns="control_success").copy()
    unknown["pitcher_id"] = 999
    unknown["batter_id"] = 888
    transformed = encoder.transform(unknown)
    assert np.isfinite(transformed.to_numpy()).all()
    assert ((transformed.filter(like="_backoff") >= 0).all()).all()


def test_backoff_prediction_is_independent_of_other_apply_rows():
    encoder = HierarchicalBackoff(alpha_leaf=10, alpha_parent=20).fit(_frame())
    apply = _frame().drop(columns="control_success")
    together = encoder.predict(apply)[0]
    alone = encoder.predict(apply.iloc[[0]])[0]
    assert np.isclose(together, alone)
