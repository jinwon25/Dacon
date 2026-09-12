import numpy as np
import pandas as pd

from src.archive.v317_pitcher_context_transport_library import (
    add_contexts,
    fit_lookup,
    map_direction,
)


def test_generic_context_lookup_preserves_unknown_pitcher() -> None:
    rows = []
    for pitcher, first_rate, second_rate in ((1, 0.8, 0.2), (2, 0.2, 0.8)):
        for balls, rate in ((0, first_rate), (3, second_rate)):
            y = [1] * int(100 * rate) + [0] * int(100 * (1.0 - rate))
            rows.extend((pitcher, balls, 0, value) for value in y)
    history = pd.DataFrame(rows, columns=["pitcher_id", "balls_before", "strikes_before", "control_success"])
    for column, value in {
        "inning": 1, "score_diff_pitcher_team": 0, "num_runners_on": 0,
        "li": 1.0, "outs_before": 0, "game_type": "R",
    }.items():
        history[column] = value
    assert "count_game_type" in add_contexts(history)
    lookup, _diag = fit_lookup(history, "count")
    query = history.iloc[:1].drop(columns=["control_success"]).copy()
    query = pd.concat([query, query.assign(pitcher_id=999)], ignore_index=True)
    direction, active = map_direction(query, np.asarray([0.5, 0.5]), "count", lookup)
    assert direction[0] > 0.0
    assert direction[1] == 0.0
    assert active.tolist() == [True, False]
