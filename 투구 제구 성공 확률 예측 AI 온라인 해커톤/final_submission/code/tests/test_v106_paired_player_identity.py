import numpy as np
import pandas as pd

from src.archive.v106_paired_player_identity import add_identity_features, optimal_eta


def test_identity_features_are_row_local_and_expected() -> None:
    rows = pd.DataFrame(
        {
            "pitcher_id": [10, 11],
            "batter_id": [20, 21],
            "batter_hand": [1, 2],
            "balls_before": [2, 0],
            "strikes_before": [1, 2],
        }
    )
    output = add_identity_features(pd.DataFrame({"x": [0.1, 0.2]}), rows)
    assert output["pitcher_count"].tolist() == ["10-2-1", "11-0-2"]
    assert output["pitcher_batter_hand"].tolist() == ["10-1", "11-2"]
    assert output["batter_count"].tolist() == ["20-2-1", "21-0-2"]


def test_optimal_eta_uses_only_paired_delta() -> None:
    axis = {
        "target": np.array([1.0, 0.0]),
        "base_parent": np.array([0.4, 0.6]),
        "direct": np.array([0.5, 0.5]),
        "domain3": np.array(["R_CORE", "R_CORE"]),
    }
    assert optimal_eta([axis], ("R_CORE",), 0.25) == 0.25
