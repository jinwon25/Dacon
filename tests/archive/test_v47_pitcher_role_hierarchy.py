import numpy as np
import pandas as pd

from src.archive.v47_pitcher_role_hierarchy import _prepare, role_bank


def test_prepare_maps_baseball_role_buckets():
    rows = pd.DataFrame(
        {
            "inning": [1, 5, 9],
            "pitcher_id": [1, 1, 2],
            "domain3": ["R_CORE", "R_CORE", "F"],
            "batter_hand": [1, 2, 1],
            "pressure": ["normal", "twostrike", "threeball"],
        }
    )
    assert _prepare(rows)["inning_bucket"].astype(str).tolist() == [
        "early",
        "middle",
        "late",
    ]


def test_role_bank_uses_only_pre_origin_seasons():
    train = pd.DataFrame(
        {
            "season": [2020, 2021, 2021, 2022, 2022],
            "inning": [1, 1, 8, 1, 8],
            "pitcher_id": [1, 1, 1, 1, 1],
            "domain3": ["R_CORE"] * 5,
            "batter_hand": [1] * 5,
            "pressure": ["normal"] * 5,
            "control_success": [0, 1, 0, 1, 1],
        }
    )
    first = role_bank(train, 2022)
    changed = train.copy()
    changed.loc[changed["season"].eq(2022), "control_success"] = 0
    second = role_bank(changed, 2022)
    for name in first:
        assert np.allclose(first[name], second[name])
