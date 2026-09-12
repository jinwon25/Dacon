from __future__ import annotations

from experiments.final_065_decision import conservative_subset_q05


def test_conservative_subset_q05_uses_weaker_scheme() -> None:
    report = {}
    for name, offset in (
        ("iid_timestamp_splits", 0.0),
        ("month_stratified_timestamp_splits", -0.1),
    ):
        report[name] = {
            split: {
                component: {"q05": 1.0 + offset}
                for component in ("score", "one_minus_nmae", "ficr")
            }
            for split in ("public", "private")
        }
    result = conservative_subset_q05(report)
    assert len(result) == 6
    assert all(value == 0.9 for value in result.values())
