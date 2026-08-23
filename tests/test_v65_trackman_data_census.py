from __future__ import annotations

import numpy as np
import pandas as pd

from src.v65_trackman_data_census import (
    derive_temporal_entity_map,
    within_group_correlation,
)


def test_temporal_map_excludes_origin_and_future_rows() -> None:
    pairs = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2022, 2022],
            "pitcher_id": [1, 1, 1, 1, 2],
            "pitcher_trackman_id": [10, 10, 10, 99, 20],
        }
    )
    mapping, audit = derive_temporal_entity_map(
        pairs, 2022, minimum_support=2, minimum_purity=0.99
    )
    assert mapping[["pitcher_id", "pitcher_trackman_id"]].to_records(index=False).tolist() == [
        (1, 10)
    ]
    assert audit["source_pairs"] == 3


def test_impure_or_low_support_mapping_is_rejected() -> None:
    pairs = pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020],
            "pitcher_id": [1, 1, 2, 2],
            "pitcher_trackman_id": [10, 11, 20, 20],
        }
    )
    mapping, _ = derive_temporal_entity_map(
        pairs, 2021, minimum_support=2, minimum_purity=0.99
    )
    assert mapping["pitcher_id"].tolist() == [2]


def test_within_group_correlation_removes_between_group_level() -> None:
    frame = pd.DataFrame(
        {
            "season": [2020] * 4,
            "pitcher_id": [1, 1, 2, 2],
            "pitch_type_group": ["fastball"] * 4,
            "feature": [0.0, 1.0, 10.0, 11.0],
            "target": [0.0, 1.0, 0.0, 1.0],
        }
    )
    value = within_group_correlation(
        frame,
        "feature",
        "target",
        ["season", "pitcher_id", "pitch_type_group"],
    )
    assert np.isclose(value, 1.0)
