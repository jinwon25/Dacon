from __future__ import annotations

import numpy as np
import pandas as pd

from src.v53_factorization_offset import _expit, _logit
from src.v86_reliability_gated_r_fm import (
    apply_reliability_offset,
    reliability_gate,
    select_source_policy,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 2, 3],
            "batter_id": [4, 5, 6],
            "pitcher_team_id": [10, 10, 11],
            "batter_team_id": [11, 12, 12],
            "balls_before": [0, 1, 2],
            "strikes_before": [0, 1, 2],
            "pitcher_hand": ["R", "R", "L"],
            "batter_hand": ["R", "L", "R"],
            "inning": [1, 5, 9],
            "base_state": ["000", "100", "111"],
            "game_type": ["R", "R", "F"],
            "asof_pitcher_n": [0, 200, 800],
            "asof_batter_n": [200, 800, 0],
            "control_success": [0, 1, 0],
            "game_month": [3, 4, 5],
        }
    )


def test_reliability_gate_is_bounded_and_pair_aware() -> None:
    rows = _rows()
    pitcher = reliability_gate(rows, "pitcher_eb200")
    geometric = reliability_gate(rows, "pair_geomean_eb200")
    minimum = reliability_gate(rows, "pair_min_eb200")
    assert np.allclose(pitcher, [0.0, 0.5, 0.8])
    assert np.allclose(geometric, [0.0, np.sqrt(0.4), 0.0])
    assert np.allclose(minimum, [0.0, 0.5, 0.0])
    assert np.all((geometric >= 0.0) & (geometric <= 1.0))


def test_apply_reliability_offset_changes_only_r_core() -> None:
    rows = _rows()
    parent = np.asarray([0.4, 0.5, 0.6])
    correction = np.asarray([0.2, -0.2, 0.2])
    candidate, active, gate = apply_reliability_offset(
        rows, parent, correction, "pitcher_eb200"
    )
    assert active.tolist() == [True, True, False]
    assert candidate[0] == parent[0]
    assert np.isclose(
        candidate[1], _expit(_logit(parent[1]) - 0.10 * 0.2 * gate[1])
    )
    assert candidate[2] == parent[2]


def test_source_selector_never_prefers_a_failing_policy() -> None:
    rows = []
    for policy, gain, month, worst in (
        ("none", 2.0, 0.50, -1.0),
        ("pitcher_eb200", 1.0, 0.75, 0.2),
        ("pair_geomean_eb200", 0.5, 1.0, 0.1),
        ("pair_min_eb200", -0.1, 1.0, 0.1),
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
    chosen, ranking = select_source_policy(pd.DataFrame(rows))
    assert chosen == "pitcher_eb200"
    assert bool(ranking.set_index("policy").loc[chosen, "source_gate_passed"])
