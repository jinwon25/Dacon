import numpy as np
import pandas as pd

from src.archive.v315_pitcher_calendar_partial_pooling import (
    fit_calendar_lookup,
    map_calendar_direction,
)


def test_calendar_lookup_is_row_local_and_unknown_is_zero() -> None:
    rows = []
    for pitcher, april, august in ((1, 0.8, 0.2), (2, 0.2, 0.8)):
        for month, rate in ((4, april), (8, august)):
            y = [1] * int(200 * rate) + [0] * int(200 * (1.0 - rate))
            rows.extend((pitcher, month, value) for value in y)
    history = pd.DataFrame(rows, columns=["pitcher_id", "game_month", "control_success"])
    lookup, diagnostics = fit_calendar_lookup(history)
    query = pd.DataFrame({"pitcher_id": [1, 999], "game_month": [4, 4]})
    direction, active = map_calendar_direction(query, np.asarray([0.5, 0.5]), lookup)
    assert diagnostics["cells"] == 4
    assert direction[0] > 0.0
    assert direction[1] == 0.0
    assert active.tolist() == [True, False]
