from __future__ import annotations

import pandas as pd

from experiments.nwp_source_ablation import source_columns
from experiments.nwp_source_router import qualifies


def test_source_columns_keep_calendar_raw_and_own_group_only() -> None:
    frame = pd.DataFrame(
        columns=[
            "hour_sin",
            "ldaps__raw__grid_1",
            "ldaps__kpx_group_1__hub_ws",
            "ldaps__kpx_group_3__hub_ws",
            "gfs__raw__grid_1",
        ]
    )
    assert source_columns(frame, "ldaps") == [
        "hour_sin",
        "ldaps__raw__grid_1",
        "ldaps__kpx_group_3__hub_ws",
    ]


def test_router_qualification_requires_strict_component_improvement() -> None:
    evaluation = {
        "delta": {"score": 0.001, "one_minus_nmae": 0.001, "ficr": 0.0},
        "months": {
            "2024-04": {
                "score": 0.001,
                "one_minus_nmae": 0.001,
                "ficr": 0.0,
            }
        },
    }
    assert not qualifies(evaluation)
    evaluation["delta"]["ficr"] = 0.001
    evaluation["months"]["2024-04"]["ficr"] = 0.001
    assert qualifies(evaluation)
