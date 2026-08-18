from __future__ import annotations

import numpy as np
import pandas as pd

from src.v51_dynamic_pitcher_state import (
    CareerState,
    dynamic_deltas,
    fit_transition,
    season_latent_states,
    select_consensus,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022, 2022],
            "pitcher_id": [1, 1, 1, 1, 1, 1],
            "control_success": [1, 1, 1, 0, 0, 0],
        }
    )


def test_transition_does_not_use_prediction_year_state() -> None:
    first, _ = season_latent_states(_history())
    rho_before, audit_before = fit_transition(first, 2022)
    changed = _history()
    changed.loc[changed["season"].eq(2022), "control_success"] = 1
    second, _ = season_latent_states(changed)
    rho_after, audit_after = fit_transition(second, 2022)
    assert rho_before == rho_after
    assert audit_before == audit_after


def test_new_pitcher_has_zero_dynamic_delta() -> None:
    states, rates = season_latent_states(_history())
    rows = pd.DataFrame(
        {
            "pitcher_id": [999],
            "asof_pitcher_n": [0.0],
            "asof_pitcher_success_rate": [0.0],
        }
    )
    empty = CareerState(pd.Series(dtype=float), pd.Series(dtype=float))
    deltas, audit = dynamic_deltas(rows, 2022, states, rates, empty)
    assert all(np.array_equal(value, np.zeros(1)) for value in deltas.values())
    assert audit["known_prior_state_rate"] == 0.0


def test_current_season_sample_downweights_prior_delta() -> None:
    history = pd.DataFrame(
        {
            "season": [2020] * 20 + [2021] * 20,
            "pitcher_id": [1] * 10 + [2] * 10 + [1] * 10 + [2] * 10,
            "control_success": (
                [1] * 8
                + [0] * 2
                + [1] * 2
                + [0] * 8
                + [1] * 8
                + [0] * 2
                + [1] * 2
                + [0] * 8
            ),
        }
    )
    states, rates = season_latent_states(history)
    career = CareerState(
        n=pd.Series([20.0], index=[1]),
        successes=pd.Series([16.0], index=[1]),
    )
    rows = pd.DataFrame(
        {
            "pitcher_id": [1, 1],
            "asof_pitcher_n": [20.0, 120.0],
            "asof_pitcher_success_rate": [0.8, 0.8],
        }
    )
    deltas, _ = dynamic_deltas(rows, 2022, states, rates, career)
    assert abs(deltas["last_k30"][1]) < abs(deltas["last_k30"][0])


def test_consensus_requires_both_selection_origins() -> None:
    common = {
        "signal": "ar_k30_w025",
        "domain": "ALL",
        "positive_month_fraction": 1.0,
        "worst_month_gain": 1.0,
        "minimum_domain_gain": 1.0,
        "applied_domain_gain": 1.0,
        "selection_score": 1.0,
        "mean_abs_shift": 0.001,
    }
    stage1 = pd.DataFrame([{**common, "gain": 2.0}])
    stage2 = pd.DataFrame([{**common, "gain": -0.1}])
    _, chosen = select_consensus(stage1, stage2)
    assert not bool(chosen["passes_consensus_gate"])
