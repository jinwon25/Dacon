import numpy as np
import pandas as pd

from src.v108_pitcher_balanced_weighting import pitcher_balance_weights


def test_pitcher_balance_is_inverse_sqrt_and_season_normalized() -> None:
    rows = pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020, 2021, 2021],
            "pitcher_id": [1, 1, 1, 2, 3, 4],
        }
    )
    weights = pitcher_balance_weights(rows, 0.5, True)
    assert np.isclose(weights[:4].mean(), 1.0)
    assert np.isclose(weights[4:].mean(), 1.0)
    assert weights[3] > weights[0]


def test_no_balancing_returns_unit_weights_after_normalization() -> None:
    rows = pd.DataFrame({"season": [2020, 2020], "pitcher_id": [1, 2]})
    assert np.allclose(pitcher_balance_weights(rows, 0.0, True), 1.0)
