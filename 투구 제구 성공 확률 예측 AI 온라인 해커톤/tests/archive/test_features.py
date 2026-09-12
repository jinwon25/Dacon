import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.archive.features import FeatureBuilder, build_trackman_context


def _main_rows():
    return pd.DataFrame(
        {
            "row_id": ["a", "b"], "season": [2024, 2024], "game_month": [4, 5],
            "game_dayofweek": [1, 2], "inning": [1, 8], "top_bottom": ["T", "B"],
            "game_type": ["R", "R"], "balls_before": [0, 3], "strikes_before": [0, 2],
            "outs_before": [0, 2], "run_top_before": [0, 2], "run_bot_before": [0, 1],
            "run_total_before": [0, 3], "score_diff_home": [0, -1],
            "score_diff_pitcher_team": [0, 1], "runner_on_1b": [0, 1],
            "runner_on_2b": [0, 0], "runner_on_3b": [0, 1], "num_runners_on": [0, 2],
            "base_state": ["___", "1_3"], "home_win_expectancy": [50.0, 35.0],
            "away_win_expectancy": [50.0, 65.0], "li": [0.8, 2.0], "pitcher_id": [1, 2],
            "batter_id": [10, 20], "pitcher_hand": [1, 2], "batter_hand": [2, 1],
            "pitcher_team_id": [1, 2], "batter_team_id": [2, 1], "asof_pitcher_n": [0, 100],
            "asof_pitcher_success_rate": [np.nan, 0.5], "asof_pitcher_reverse_rate": [np.nan, 0.2],
            "asof_pitcher_middle_rate": [np.nan, 0.2], "asof_pitcher_ball_rate": [np.nan, 0.4],
            "asof_pitcher_strike_rate": [np.nan, 0.4],
            "asof_pitcher_prev1_game_success_rate": [np.nan, 0.4],
            "asof_pitcher_prev3_game_success_rate": [np.nan, 0.5],
            "asof_pitcher_prev5_game_success_rate": [np.nan, 0.55],
            "asof_pitcher_prev1_game_middle_rate": [np.nan, 0.2],
            "asof_pitcher_prev3_game_middle_rate": [np.nan, 0.2],
            "asof_pitcher_prev5_game_middle_rate": [np.nan, 0.2], "asof_batter_n": [0, 200],
            "asof_batter_success_rate": [np.nan, 0.52], "asof_batter_middle_rate": [np.nan, 0.2],
            "asof_pitcher_pitchmix_n": [0, 100], "asof_pitcher_fastball_rate": [np.nan, 0.5],
            "asof_pitcher_breaking_rate": [np.nan, 0.3], "asof_pitcher_offspeed_rate": [np.nan, 0.2],
        }
    )


def test_transform_is_independent_of_other_test_rows():
    rows = _main_rows()
    builder = FeatureBuilder("engineered").fit(rows, np.array([0, 1]))
    together = builder.transform(rows).iloc[[0]].reset_index(drop=True)
    alone = builder.transform(rows.iloc[[0]]).reset_index(drop=True)
    assert_frame_equal(together, alone)


def test_unknown_categories_use_minus_one():
    rows = _main_rows()
    builder = FeatureBuilder("official").fit(rows.iloc[[0]], np.array([0]))
    transformed = builder.transform(rows.iloc[[1]])
    assert transformed["pitcher_id"].iloc[0] == -1


def test_trackman_context_uses_only_prior_seasons():
    trackman = pd.DataFrame(
        {
            "season": [2022, 2023], "balls_before": [0, 0], "strikes_before": [0, 0],
            "outs_before": [0, 0], "pitch_type_group": ["fastball", "breaking"],
            **{measure: [1.0, 3.0] for measure in [
                "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
                "rel_height", "rel_side", "zone_speed"
            ]},
        }
    )
    context = build_trackman_context(trackman)
    value_2023 = context.loc[context["season"] == 2023, "tm_rel_speed_mean"].iloc[0]
    value_2024 = context.loc[context["season"] == 2024, "tm_rel_speed_mean"].iloc[0]
    assert value_2023 == 1.0
    assert value_2024 == 2.0
