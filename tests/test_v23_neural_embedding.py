import numpy as np
import pandas as pd

from src.v23_neural_embedding_screen import (
    _category_codes,
    _diagnostics,
    _numeric_columns,
    prepare,
)


def test_unseen_category_maps_to_zero():
    fit, audit, cardinality = _category_codes(
        pd.Series(["a", "b", "a"]), pd.Series(["b", "c"])
    )
    assert fit.tolist() == [1, 2, 1]
    assert audit.tolist() == [2, 0]
    assert cardinality == 3


def test_numeric_columns_exclude_ids_labels_and_categories():
    frame = pd.DataFrame(
        {
            "row_id": [1],
            "season": [2022],
            "control_success": [1],
            "pitcher_id": [10],
            "domain3": ["R_CORE"],
            "li": [1.2],
            "asof_pitcher_n": [50],
        }
    )
    assert _numeric_columns(frame) == ["li", "asof_pitcher_n"]


def test_domain_route_leaves_other_domains_unchanged():
    frame = pd.DataFrame(
        {
            "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "v22": [0.5] * 6,
            "game_month": [3, 3, 4, 4, 5, 5],
            "domain3": ["R_CORE", "R_ANCHOR", "F", "R_CORE", "R_ANCHOR", "F"],
        }
    )
    direct = frame["target"].to_numpy()
    result = _diagnostics(frame, direct, 0.5, ("R_CORE",))
    assert result["r_core_gain"] > 0.0
    assert result["r_anchor_gain"] == 0.0
    assert result["f_gain"] == 0.0


def test_group_equal_weights_balance_season_domain_cells():
    rows = []
    for season, domain, count in [(2020, "R_CORE", 2), (2020, "F", 1), (2021, "R_CORE", 4)]:
        for index in range(count):
            rows.append(
                {
                    "season": season,
                    "domain3": domain,
                    "control_success": index % 2,
                    "game_dayofweek": "월",
                    "top_bottom": "초",
                    "game_type": "R",
                    "base_state": 0,
                    "pitcher_hand": "R",
                    "batter_hand": "L",
                    "pitcher_team_id": 1,
                    "batter_team_id": 2,
                    "pitcher_id": 3,
                    "batter_id": 4,
                    "count_state": "0-0",
                    "hand_matchup": "R-L",
                    "inning_bucket": "early",
                    "li": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    prepared = prepare(frame, frame.iloc[:1], weighting="season_domain_equal")
    key = frame["season"].astype(str) + "-" + frame["domain3"]
    totals = pd.Series(prepared.weight_fit).groupby(key).sum().to_numpy()
    np.testing.assert_allclose(totals, np.repeat(totals[0], len(totals)))
