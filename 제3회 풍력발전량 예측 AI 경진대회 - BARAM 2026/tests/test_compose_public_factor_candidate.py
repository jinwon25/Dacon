from __future__ import annotations

import pandas as pd
import pytest

from experiments.compose_public_factor_candidate import (
    compose_identified_factor,
)


def _frame(
    group1: list[float],
    group2: list[float],
    group3: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": [
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
            ],
            "kpx_group_1": group1,
            "kpx_group_2": group2,
            "kpx_group_3": group3,
        }
    )


def test_compose_identified_factor_transfers_only_target() -> None:
    base = _frame([10.0, 20.0], [30.0, 40.0], [50.0, 60.0])
    control = _frame([10.0, 20.0], [31.0, 41.0], [51.0, 61.0])
    treatment = _frame([12.0, 18.0], [31.0, 41.0], [51.0, 61.0])

    result = compose_identified_factor(
        base,
        treatment,
        control,
        target="kpx_group_1",
    )

    assert result["kpx_group_1"].tolist() == [12.0, 18.0]
    assert result["kpx_group_2"].tolist() == [30.0, 40.0]
    assert result["kpx_group_3"].tolist() == [50.0, 60.0]


def test_compose_identified_factor_rejects_non_target_difference() -> None:
    base = _frame([10.0, 20.0], [30.0, 40.0], [50.0, 60.0])
    control = _frame([10.0, 20.0], [31.0, 41.0], [51.0, 61.0])
    treatment = _frame([12.0, 18.0], [32.0, 41.0], [51.0, 61.0])

    with pytest.raises(ValueError, match="differ outside"):
        compose_identified_factor(
            base,
            treatment,
            control,
            target="kpx_group_1",
        )


def test_compose_identified_factor_rejects_unmatched_base_target() -> None:
    base = _frame([9.0, 20.0], [30.0, 40.0], [50.0, 60.0])
    control = _frame([10.0, 20.0], [31.0, 41.0], [51.0, 61.0])
    treatment = _frame([12.0, 18.0], [31.0, 41.0], [51.0, 61.0])

    with pytest.raises(ValueError, match="base target vector"):
        compose_identified_factor(
            base,
            treatment,
            control,
            target="kpx_group_1",
        )
