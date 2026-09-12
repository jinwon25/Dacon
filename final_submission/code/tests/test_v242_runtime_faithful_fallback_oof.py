import numpy as np
import pandas as pd

from src.archive.v242_runtime_faithful_fallback_oof import build_frozen_lookup


def test_lookup_uses_only_supplied_history_and_last_season_anchor(monkeypatch):
    history = pd.DataFrame(
        {
            "season": [2020, 2021, 2021],
            "pitcher_id": [1, 1, 1],
            "batter_id": [2, 2, 3],
            "control_success": [0, 1, 1],
            "top_bottom": ["T", "T", "B"],
            "game_type": ["R", "R", "R"],
            "base_state": ["___", "___", "1__"],
            "pitcher_hand": [2, 2, 2],
            "batter_hand": [1, 1, 2],
            "pitcher_team_id": [1, 1, 1],
            "batter_team_id": [2, 2, 3],
            "asof_pitcher_n": [10, 20, 30],
            "asof_pitcher_success_rate": [0.4, 0.5, 0.6],
            "asof_pitcher_reverse_rate": [0.1, 0.1, 0.1],
            "asof_pitcher_middle_rate": [0.2, 0.2, 0.2],
            "asof_pitcher_ball_rate": [0.3, 0.3, 0.3],
            "asof_pitcher_strike_rate": [0.4, 0.4, 0.4],
            "asof_batter_n": [5, 7, 9],
            "asof_batter_success_rate": [0.4, 0.5, 0.6],
            "asof_batter_middle_rate": [0.2, 0.2, 0.2],
            "balls_before": [0, 3, 0],
            "strikes_before": [0, 0, 2],
            "runner_on_1b": [0, 0, 1],
            "runner_on_2b": [0, 0, 0],
            "runner_on_3b": [0, 0, 0],
            "inning": [1, 2, 1],
            "li": [1.0, 1.0, 1.0],
            "score_diff_pitcher_team": [0, 0, 0],
        }
    )
    monkeypatch.setattr(
        "src.archive.v242_runtime_faithful_fallback_oof._appearance_table",
        lambda frame: pd.DataFrame(
            {"pitcher_id": [1], "season": [2021], "ppa": [12.0]}
        ),
    )
    monkeypatch.setattr(
        "src.archive.v242_runtime_faithful_fallback_oof._trackman_season_profiles",
        lambda trackman, pitcher_map: pd.DataFrame(
            {
                "pitcher_id": [1],
                "season": [2021],
                "rel_speed": [140.0],
                "rel_speed_sd": [2.0],
                "spin_rate": [2200.0],
                "induced_vert_break": [30.0],
                "horz_break": [5.0],
                "extension": [1.8],
                "rel_height": [1.7],
                "rel_side": [-0.5],
            }
        ),
    )
    lookup = build_frozen_lookup(history, pd.DataFrame(), pd.DataFrame())

    # The latest season's minimum counter (20), not the full-career minimum.
    assert lookup["anchors"]["p_succ"][1][0] == 20.0
    assert np.isclose(lookup["target_mean"], 2.0 / 3.0)
    assert lookup["ppa"][1] == 12.0
    assert lookup["tm"]["rel_speed"][1] == 140.0
