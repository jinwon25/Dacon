from __future__ import annotations

import numpy as np
import pandas as pd

from src.champion.v53_factorization_offset import _expit, _logit
from src.archive.v89_reliability_gated_f_extra_dose import apply_extra_dose, select_recipe


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "batter_id": [3, 4],
            "pitcher_team_id": [10, 10],
            "batter_team_id": [11, 12],
            "balls_before": [0, 1],
            "strikes_before": [0, 1],
            "pitcher_hand": ["R", "L"],
            "batter_hand": ["L", "R"],
            "inning": [1, 9],
            "base_state": ["000", "100"],
            "game_type": ["F", "R"],
            "asof_pitcher_n": [200, 800],
            "asof_batter_n": [800, 200],
            "control_success": [1, 0],
            "game_month": [3, 4],
        }
    )


def test_extra_dose_changes_only_f_and_uses_row_reliability() -> None:
    frame = _rows()
    base = np.asarray([0.5, 0.5])
    correction = np.asarray([0.2, 0.2])
    candidate, active, gate = apply_extra_dose(
        frame, base, correction, "pitcher_eb200", 0.05
    )
    assert active.tolist() == [True, False]
    assert np.isclose(candidate[0], _expit(_logit(base[0]) + 0.05 * 0.5 * 0.2))
    assert candidate[1] == base[1]
    assert np.all((gate >= 0.0) & (gate <= 1.0))


def test_selector_rejects_high_gain_unstable_recipe() -> None:
    recipes = (
        ("uniform_d0.025", "uniform", 0.025, 2.0, 0.5),
        ("pitcher_eb200_d0.025", "pitcher_eb200", 0.025, 1.0, 0.75),
        ("pair_geomean_eb200_d0.025", "pair_geomean_eb200", 0.025, 0.5, 1.0),
        ("pitcher_eb200_d0.050", "pitcher_eb200", 0.05, -0.1, 1.0),
        ("pair_geomean_eb200_d0.050", "pair_geomean_eb200", 0.05, -0.1, 1.0),
    )
    rows = []
    for name, policy, delta, gain, month in recipes:
        for axis in ("full_2022", "late_2023"):
            rows.append(
                {
                    "recipe": name,
                    "policy": policy,
                    "delta": delta,
                    "axis": axis,
                    "gain": gain,
                    "positive_month_fraction": month,
                    "worst_month_gain": 0.1,
                    "minimum_domain_gain": 0.0,
                    "applied_domain_gain": gain,
                }
            )
    selected, ranking = select_recipe(pd.DataFrame(rows))
    assert selected["recipe"] == "pitcher_eb200_d0.025"
    assert bool(selected["source_gate_passed"])
