from __future__ import annotations

import numpy as np
import pandas as pd

from src.v87_cross_season_consensus_r_fm import (
    combine_corrections,
    select_policy,
)


def test_consensus_combiners_are_row_local_and_bounded() -> None:
    older = np.asarray([0.2, -0.2, 0.1, -0.4])
    recent = np.asarray([0.1, -0.3, -0.2, 0.2])
    sign_mean, agree = combine_corrections(older, recent, "sign_mean")
    sign_min, _ = combine_corrections(older, recent, "sign_min")
    quarter, _ = combine_corrections(older, recent, "disagreement_quarter")
    assert agree.tolist() == [True, True, False, False]
    assert np.allclose(sign_mean, [0.15, -0.25, 0.0, 0.0])
    assert np.allclose(sign_min, [0.1, -0.2, 0.0, 0.0])
    assert np.allclose(quarter[:3], [0.15, -0.25, -0.0125])
    assert np.max(np.abs(quarter)) <= 0.25


def test_selector_uses_only_source_gate_passers() -> None:
    rows = []
    for policy, gain, month, worst in (
        ("source_mean", 3.0, 0.50, -2.0),
        ("sign_mean", 1.5, 0.75, 0.3),
        ("sign_min", 1.0, 1.0, 0.2),
        ("disagreement_quarter", -0.1, 1.0, 0.2),
    ):
        for axis in ("full_2022", "late_2023"):
            rows.append(
                {
                    "policy": policy,
                    "axis": axis,
                    "gain": gain,
                    "positive_month_fraction": month,
                    "worst_month_gain": worst,
                    "minimum_domain_gain": 0.0,
                    "applied_domain_gain": gain,
                }
            )
    chosen, ranking = select_policy(pd.DataFrame(rows))
    assert chosen == "sign_mean"
    assert bool(ranking.set_index("policy").loc[chosen, "source_gate_passed"])
