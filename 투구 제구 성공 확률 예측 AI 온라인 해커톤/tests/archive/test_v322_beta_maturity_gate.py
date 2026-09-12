import numpy as np
import pandas as pd

from src.archive.v322_beta_maturity_gate import maturity_gates


def test_maturity_gates_are_row_local_and_bounded():
    history = pd.DataFrame(
        {
            "season": [2021],
            "pitcher_id": [3],
            "asof_pitcher_n": [9.0],
            "asof_pitcher_success_rate": [0.5],
            "control_success": [1.0],
        }
    )
    frame = pd.DataFrame(
        {
            "season": [2022, 2022],
            "game_month": [6, 9],
            "pitcher_id": [3, 3],
            "asof_pitcher_n": [110.0, 710.0],
            "asof_pitcher_success_rate": [0.5, 0.5],
            "control_success": [0.0, 1.0],
        }
    )
    train = pd.concat([history, frame], ignore_index=True)
    gates = maturity_gates(frame, train, 2022)
    assert set(gates) == {
        "all",
        "linear_n200",
        "linear_n500",
        "pitcher_n_ge100",
        "pitcher_n_ge300",
        "pitcher_n_ge600",
        "month_ge7",
        "month_ge8",
        "month_ge9",
    }
    assert all(np.all((value >= 0.0) & (value <= 1.0)) for value in gates.values())
    np.testing.assert_array_equal(gates["month_ge9"], [0.0, 1.0])
