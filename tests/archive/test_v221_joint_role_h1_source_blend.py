from __future__ import annotations

from src.archive.v221_joint_role_h1_source_blend import (
    restrictions,
    select_role_weight,
)


def _item(gain: float, fraction: float = 1.0, worst: float = 1.0):
    return {
        "gain": gain,
        "positive_month_fraction": fraction,
        "worst_month_gain": worst,
    }


def test_select_role_weight_maximizes_worst_source_gain() -> None:
    results = {
        "0.00": {"2022": _item(5.0), "2023": _item(6.0)},
        "0.50": {"2022": _item(5.5), "2023": _item(8.0)},
        "1.00": {"2022": _item(6.0), "2023": _item(9.0, worst=-5.1)},
    }
    assert select_role_weight(results) == 0.5


def test_v221_locks_2024() -> None:
    audit = restrictions()
    assert audit["source_only_weight_selection"]
    assert audit["locked_2024_not_used_for_selection"]
    assert audit["no_new_model_fit"]
    assert not audit["test_csv_read"]
