import numpy as np
import pandas as pd
import torch

from src.v45_tabm_mini_screen import TabMMini, apply_direct, recent_history


def test_tabm_mini_outputs_one_probability_per_member():
    model = TabMMini(3, [4, 5], k=6, width=8, dropout=0.0)
    output = model(
        torch.zeros((7, 3)),
        torch.column_stack(
            [torch.arange(7) % 4, torch.arange(7) % 5]
        ),
    )
    assert output.shape == (7, 6)
    assert torch.all((output > 0.0) & (output < 1.0))


def test_recent_history_uses_exactly_two_latest_seasons():
    rows = []
    for year in (2020, 2021, 2022):
        rows.append(
            {
                "season": year,
                "game_type": "R",
                "pitcher_team_id": 1,
                "batter_team_id": 2,
                "balls_before": 0,
                "strikes_before": 0,
                "pitcher_hand": 1,
                "batter_hand": 2,
                "inning": 1,
                "asof_pitcher_n": 0,
                "asof_batter_n": 0,
                "asof_pitcher_pitchmix_n": 0,
                "asof_pitcher_prev1_game_success_rate": np.nan,
                "asof_pitcher_prev5_game_success_rate": np.nan,
                "asof_pitcher_prev1_game_middle_rate": np.nan,
                "asof_pitcher_prev5_game_middle_rate": np.nan,
                "asof_pitcher_success_rate": np.nan,
                "asof_batter_success_rate": np.nan,
                "control_success": 0.0,
            }
        )
    output = recent_history(pd.DataFrame(rows), 2022)
    assert output["season"].tolist() == [2021, 2022]


def test_apply_direct_only_changes_routed_domain():
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.5, 0.6],
            "v25": [0.4, 0.5, 0.6],
            "domain3": ["R_CORE", "R_ANCHOR", "F"],
        }
    )
    candidate, active = apply_direct(
        frame, np.array([0.8, 0.8, 0.8]), "R_CORE", 0.1
    )
    np.testing.assert_array_equal(active, [True, False, False])
    np.testing.assert_allclose(candidate, [0.44, 0.5, 0.6])
