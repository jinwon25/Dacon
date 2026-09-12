from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.compose_issue_trajectory_tcn_candidate import (
    compose_single_group_factor,
)


def test_compose_single_group_factor_changes_only_target() -> None:
    incumbent = pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": ["t1", "t2"],
            "kpx_group_1": [10.0, 20.0],
            "kpx_group_2": [30.0, 40.0],
            "kpx_group_3": [100.0, 200.0],
        }
    )
    output = compose_single_group_factor(
        incumbent,
        np.array([300.0, 0.0]),
        target="kpx_group_3",
        weight=0.05,
    )
    assert output["kpx_group_1"].equals(incumbent["kpx_group_1"])
    assert output["kpx_group_2"].equals(incumbent["kpx_group_2"])
    assert np.allclose(output["kpx_group_3"], [110.0, 190.0])


def test_compose_single_group_factor_checks_alignment() -> None:
    incumbent = pd.DataFrame(
        {
            "kpx_group_1": [1.0],
            "kpx_group_2": [1.0],
            "kpx_group_3": [1.0],
        }
    )
    try:
        compose_single_group_factor(
            incumbent,
            np.array([1.0, 2.0]),
            target="kpx_group_3",
            weight=0.05,
        )
    except ValueError as error:
        assert "row counts differ" in str(error)
    else:
        raise AssertionError("alignment error was not raised")
