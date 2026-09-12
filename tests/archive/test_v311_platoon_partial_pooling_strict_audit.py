import numpy as np
import pandas as pd

from src.archive.v311_platoon_partial_pooling_strict_audit import (
    fit_platoon_lookup,
    map_platoon_direction,
)


def test_platoon_lookup_is_row_local_and_preserves_unknown() -> None:
    rows = []
    for pitcher, left_rate, right_rate in ((1, 0.7, 0.3), (2, 0.3, 0.7)):
        for hand, rate in ((1, left_rate), (2, right_rate)):
            outcomes = [1] * int(100 * rate) + [0] * int(100 * (1 - rate))
            rows.extend((pitcher, hand, outcome) for outcome in outcomes)
    history = pd.DataFrame(rows, columns=["pitcher_id", "batter_hand", "control_success"])
    lookup, diagnostics = fit_platoon_lookup(history)
    query = pd.DataFrame({"pitcher_id": [1, 999], "batter_hand": [1, 1]})
    direction, active = map_platoon_direction(query, np.asarray([0.5, 0.5]), lookup)
    assert diagnostics["cells"] == 4
    assert direction[0] > 0.0
    assert direction[1] == 0.0
    assert active.tolist() == [True, False]
