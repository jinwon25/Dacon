import numpy as np
import pandas as pd

from src.archive.v37_latest_season_catboost_residual import (
    _cat_frame,
    apply_correction,
    categorical_columns,
)


def _raw():
    return pd.DataFrame(
        {
            "game_dayofweek": [1],
            "top_bottom": ["T"],
            "game_type": ["R"],
            "domain3": ["R_CORE"],
            "base_state": ["000"],
            "pitcher_hand": [1],
            "batter_hand": [2],
            "pitcher_team_id": [1],
            "batter_team_id": [2],
            "balls_before": [1],
            "strikes_before": [2],
            "pitcher_id": [10],
            "batter_id": [20],
        }
    )


def test_player_variant_adds_ids_but_context_does_not():
    assert "pitcher_id" not in categorical_columns("context")
    assert "pitcher_id" in categorical_columns("player_ids")
    context = _cat_frame(pd.DataFrame({"x": [1.0]}), _raw(), np.array([0.4]), "context")
    player = _cat_frame(
        pd.DataFrame({"x": [1.0]}), _raw(), np.array([0.4]), "player_ids"
    )
    assert "pitcher_id" not in context
    assert player.loc[0, "pitcher_id"] == "10"


def test_apply_correction_protects_other_domains():
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.4],
            "v25": [0.4, 0.4],
            "domain3": ["R_CORE", "F"],
        }
    )
    candidate, active = apply_correction(
        frame, np.array([0.05, 0.05]), eta=0.2, domain="R_CORE"
    )
    assert np.allclose(candidate, [0.41, 0.4])
    assert active.tolist() == [True, False]
