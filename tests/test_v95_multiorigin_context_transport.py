from __future__ import annotations

import numpy as np
import pandas as pd

from src.v95_multiorigin_context_transport import (
    add_context_features,
    apply_effect,
    cache_group_keys,
    domain_group_keys,
    fit_centered_effect,
    group_keys,
    map_effect,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "balls_before": [0, 3, 1, 0], "strikes_before": [0, 1, 2, 0],
            "pitcher_hand": ["R", "L", "R", "L"], "batter_hand": ["R", "R", "L", "L"],
            "inning": [1, 4, 7, 9], "num_runners_on": [0, 1, 2, 0],
            "score_diff_pitcher_team": [0, -2, 4, 0], "li": [0.5, 1.0, 4.0, 0.8],
            "game_type": ["R", "R", "F", "R"], "base_state": ["0", "1", "12", "0"],
            "outs_before": [0, 1, 2, 0], "pitcher_team_id": [1, 1, 2, 2],
            "batter_team_id": [2, 2, 1, 1], "domain3": ["R_CORE", "R_CORE", "F", "R_ANCHOR"],
            "control_success": [1, 0, 1, 0],
        }
    )


def test_context_features_are_row_order_equivariant() -> None:
    frame = _frame()
    original = add_context_features(frame)
    order = np.array([2, 0, 3, 1])
    shuffled = add_context_features(frame.iloc[order])
    for column in ("count_state", "hand_matchup", "pressure", "inning_bucket", "runner_band", "score_bucket", "leverage_bucket"):
        np.testing.assert_array_equal(shuffled[column], original[column].to_numpy()[order])


def test_group_keys_are_row_local() -> None:
    frame = add_context_features(_frame())
    original = group_keys(frame, ("count_state", "hand_matchup"))
    np.testing.assert_array_equal(group_keys(frame.iloc[::-1], ("count_state", "hand_matchup")), original[::-1])


def test_cached_group_keys_equal_uncached_keys() -> None:
    frame = add_context_features(_frame())
    cached = cache_group_keys(frame)
    first = domain_group_keys(cached, ("count_state", "hand_matchup"))
    second = domain_group_keys(cache_group_keys(frame.copy()), ("count_state", "hand_matchup"))
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint64


def test_unknown_effect_maps_to_zero() -> None:
    frame = add_context_features(_frame())
    mapped = map_effect(frame, ("count_state",), {"R_CORE\x1e0-0": 0.01})
    np.testing.assert_allclose(mapped, [0.01, 0.0, 0.0, 0.0])


def test_centered_effect_and_application_respect_route() -> None:
    frame = add_context_features(_frame())
    parent = np.full(len(frame), 0.5)
    effect = fit_centered_effect(frame, parent, ("count_state",), ("R_CORE",), alpha=1.0)
    correction = map_effect(frame, ("count_state",), effect)
    output, active = apply_effect(frame, parent, correction, ("R_CORE",), eta=1.0)
    np.testing.assert_array_equal(active, [True, True, False, False])
    np.testing.assert_allclose(output[~active], 0.5)
    assert np.all((output >= 0.001) & (output <= 0.999))
