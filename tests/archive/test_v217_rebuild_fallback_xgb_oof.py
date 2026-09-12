from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v217_rebuild_fallback_xgb_oof import (
    _anchor_table,
    restrictions,
    situation_masks,
)


def test_anchor_table_takes_minimum_count_within_id_season() -> None:
    frame = pd.DataFrame({
        "pitcher_id": [1, 1, 1],
        "season": [2022, 2022, 2023],
        "asof_pitcher_n": [20, 10, 40],
        "asof_pitcher_success_rate": [0.6, 0.5, 0.55],
    })
    anchor = _anchor_table(
        frame,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    assert anchor["asof_pitcher_n"].tolist() == [10, 40]
    np.testing.assert_allclose(anchor["succ"], [5.0, 22.0])


def test_situation_masks_keep_public_hi_li_strictness() -> None:
    frame = pd.DataFrame({
        "balls_before": [3], "strikes_before": [2],
        "runner_on_1b": [0], "runner_on_2b": [1], "runner_on_3b": [0],
        "batter_hand": [1], "inning": [7], "li": [1.5],
        "score_diff_pitcher_team": [5],
    })
    masks = situation_masks(frame)
    assert masks["3ball"][0]
    assert masks["2strk"][0]
    assert masks["risp"][0]
    assert not masks["hiLI"][0]
    assert masks["blowout"][0]


def test_v217_restrictions_are_strict_and_test_independent() -> None:
    audit = restrictions()
    assert audit["strictly_prior_season_label_lookups"]
    assert audit["strictly_prior_season_trackman_profiles"]
    assert audit["fixed_public1175_recipe"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["public_score_used_for_selection"]
