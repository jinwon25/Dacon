from __future__ import annotations

import pandas as pd
import pytest

from experiments.compose_single_target_probe import compose_single_target


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": [1, 2],
            "forecast_kst_dtm": ["2025-01-01 01:00", "2025-01-01 02:00"],
            "kpx_group_1": [10.0, 11.0],
            "kpx_group_2": [20.0, 21.0],
            "kpx_group_3": [30.0, 31.0],
        }
    )


def test_compose_single_target_changes_only_requested_column() -> None:
    incumbent = _frame()
    source = _frame()
    source["kpx_group_1"] += 100.0
    source["kpx_group_3"] += 3.0

    output = compose_single_target(
        incumbent, source, target="kpx_group_3"
    )

    assert output["kpx_group_1"].equals(incumbent["kpx_group_1"])
    assert output["kpx_group_2"].equals(incumbent["kpx_group_2"])
    assert output["kpx_group_3"].equals(source["kpx_group_3"])


def test_compose_single_target_rejects_misaligned_ids() -> None:
    incumbent = _frame()
    source = _frame()
    source.loc[0, "forecast_id"] = 99

    with pytest.raises(ValueError, match="identifiers differ"):
        compose_single_target(
            incumbent, source, target="kpx_group_3"
        )
