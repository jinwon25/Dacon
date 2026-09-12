import numpy as np
import pandas as pd

from src.archive.v313_recent_f_lowrank_interaction import (
    additive_source_parent,
    analytic_dose,
    fit_direction,
)


def _rows() -> pd.DataFrame:
    rows = []
    for pitcher in (1, 2):
        for balls in range(4):
            for strikes in range(3):
                for hand in (1, 2):
                    for repeat in range(20):
                        success = int((pitcher == 1) == (hand == 1))
                        rows.append((pitcher, balls, strikes, hand, success))
    return pd.DataFrame(
        rows,
        columns=["pitcher_id", "balls_before", "strikes_before", "batter_hand", "control_success"],
    )


def test_direction_is_row_local_and_unknown_pitcher_is_zero() -> None:
    source = _rows()
    parent = additive_source_parent(source, source["control_success"].to_numpy())
    assert len(parent) == len(source)
    query = pd.DataFrame(
        {
            "pitcher_id": [1, 999],
            "balls_before": [0, 0],
            "strikes_before": [0, 0],
            "batter_hand": [1, 1],
        }
    )
    direction, seen, _diagnostics = fit_direction(source, query)
    assert seen.tolist() == [True, False]
    assert direction[0] != 0.0
    assert direction[1] == 0.0


def test_analytic_dose_is_bounded() -> None:
    target = np.asarray([1.0, 0.0])
    parent = np.asarray([0.5, 0.5])
    direction = np.asarray([0.1, -0.1])
    active = np.asarray([True, True])
    assert analytic_dose(target, parent, direction, active) == 1.0
