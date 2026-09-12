import numpy as np
import pandas as pd

from src.archive.v325_within_regime_residual_calibrator import (
    add_keys,
    fit_lookup,
    map_lookup,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_type": ["F", "F", "F", "F"],
            "balls_before": [0, 0, 3, 3],
            "strikes_before": [0, 0, 2, 2],
            "pitcher_hand": [1, 1, 2, 2],
            "batter_hand": [1, 2, 1, 2],
            "control_success": [1, 0, 1, 0],
        }
    )


def test_keys_are_row_local_and_shuffle_equivariant() -> None:
    frame = _frame()
    parent = np.asarray([0.45, 0.46, 0.55, 0.54])
    expected = add_keys(frame, parent)
    order = np.asarray([2, 0, 3, 1])
    shuffled = add_keys(frame.iloc[order].reset_index(drop=True), parent[order])
    pd.testing.assert_frame_equal(shuffled, expected.iloc[order].reset_index(drop=True))


def test_lookup_maps_seen_groups_and_zeroes_unseen_groups() -> None:
    frame = _frame()
    parent = np.full(4, 0.5)
    keys = ("game_type", "count")
    lookup = fit_lookup(frame, parent, keys, alpha=1.0)
    query = frame.iloc[[0, 2]].reset_index(drop=True)
    correction, active = map_lookup(query, np.full(2, 0.5), keys, lookup)
    assert active.tolist() == [True, True]
    assert np.isfinite(correction).all()
    unseen = query.copy()
    unseen.loc[0, "balls_before"] = 2
    unseen_correction, unseen_active = map_lookup(
        unseen, np.full(2, 0.5), keys, lookup
    )
    assert not unseen_active[0]
    assert unseen_correction[0] == 0.0
