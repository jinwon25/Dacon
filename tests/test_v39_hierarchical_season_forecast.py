import numpy as np
import pandas as pd

from src.v39_hierarchical_season_forecast import (
    _player_forecasts,
    current_season_counts,
    forecast_bank,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022],
            "game_month": [4, 5, 4, 5],
            "game_type": ["R"] * 4,
            "pitcher_team_id": [1] * 4,
            "batter_team_id": [2] * 4,
            "pitcher_id": [10] * 4,
            "batter_id": [20] * 4,
            "asof_pitcher_n": [0, 1, 2, 3],
            "asof_pitcher_success_rate": [0.0, 1.0, 0.5, 2.0 / 3.0],
            "asof_batter_n": [0, 1, 2, 3],
            "asof_batter_success_rate": [0.0, 1.0, 0.5, 2.0 / 3.0],
            "control_success": [1.0, 0.0, 1.0, 0.0],
        }
    )


def test_current_season_counts_subtract_only_frozen_history():
    rows = _rows()
    history = rows.loc[rows["season"].lt(2022)]
    query = rows.loc[rows["season"].eq(2022)]
    n, success = current_season_counts(history, query, "pitcher_id")
    np.testing.assert_array_equal(n, [0.0, 1.0])
    np.testing.assert_array_equal(success, [0.0, 1.0])


def test_player_forecast_ignores_query_targets():
    rows = _rows()
    rows["domain3"] = "R_CORE"
    history = rows.loc[rows["season"].lt(2022)].copy()
    query = rows.loc[rows["season"].eq(2022)].copy()
    first = _player_forecasts(history, query, "pitcher_id", "domain")
    query["control_success"] = 1.0 - query["control_success"]
    second = _player_forecasts(history, query, "pitcher_id", "domain")
    for name in first:
        np.testing.assert_allclose(first[name], second[name])


def test_forecast_bank_is_finite_and_has_fixed_low_degree_grid():
    rows = _rows()
    rows["domain3"] = "R_CORE"
    bank = forecast_bank(rows, 2022)
    assert len(bank) == 54
    assert all(len(value) == 2 for value in bank.values())
    assert all(np.isfinite(value).all() for value in bank.values())
    assert all(((value > 0.0) & (value < 1.0)).all() for value in bank.values())
