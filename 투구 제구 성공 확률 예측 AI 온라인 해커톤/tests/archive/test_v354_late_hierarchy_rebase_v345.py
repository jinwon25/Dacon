import numpy as np
import pandas as pd

from src.archive.v354_late_hierarchy_rebase_v345 import (
    blend_toward_hierarchy,
    late_anchor_mask,
)


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_type": ["R", "R", "R", "F"],
            "pitcher_team_id": [13, 13, 16, 13],
            "batter_team_id": [16, 16, 13, 16],
            "game_month": [7, 8, 9, 9],
        }
    )


def test_late_anchor_mask_requires_regular_anchor_and_august():
    np.testing.assert_array_equal(
        late_anchor_mask(fixture()), [False, True, True, False]
    )


def test_blend_toward_hierarchy_protects_inactive_rows():
    parent = np.asarray([0.2, 0.3, 0.4, 0.5])
    hierarchy = np.asarray([0.8, 0.8, 0.0, 0.0])
    candidate, active = blend_toward_hierarchy(fixture(), parent, hierarchy)
    np.testing.assert_array_equal(active, [False, True, True, False])
    np.testing.assert_allclose(candidate, [0.2, 0.4, 0.32, 0.5])
