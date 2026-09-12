import numpy as np
import pandas as pd

from src.archive.v321_strict_beta_binomial_complement import (
    beta_candidate_matrix,
    current_season_state,
)


def test_current_season_state_differences_exact_terminal_snapshot():
    frame = pd.DataFrame(
        {
            "pitcher_id": [7, 7],
            "asof_pitcher_n": [12.0, 13.0],
            "asof_pitcher_success_rate": [7.0 / 12.0, 8.0 / 13.0],
        }
    )
    n, successes = current_season_state(
        frame,
        {7: (10.0, 6.0)},
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    np.testing.assert_allclose(n, [2.0, 3.0])
    np.testing.assert_allclose(successes, [1.0, 2.0])


def test_prediction_year_targets_do_not_enter_beta_candidates():
    rows = []
    for season in (2020, 2021):
        for index in range(3):
            rows.append(
                {
                    "season": season,
                    "game_type": "R",
                    "pitcher_id": 1,
                    "batter_id": 2,
                    "asof_pitcher_n": float((season - 2020) * 3 + index),
                    "asof_pitcher_success_rate": 0.5,
                    "asof_batter_n": float((season - 2020) * 3 + index),
                    "asof_batter_success_rate": 0.5,
                    "asof_pitcher_prev5_game_success_rate": 0.5,
                    "control_success": float(index % 2),
                }
            )
    train = pd.DataFrame(rows)
    _, original = beta_candidate_matrix(train, 2021, 20.0)
    changed = train.copy()
    changed.loc[changed["season"].eq(2021), "control_success"] = 1.0
    _, mutated = beta_candidate_matrix(changed, 2021, 20.0)
    np.testing.assert_allclose(original, mutated)
