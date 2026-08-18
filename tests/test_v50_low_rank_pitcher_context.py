from __future__ import annotations

import numpy as np
import pandas as pd

from src.v50_low_rank_pitcher_context import (
    build_audit_bank,
    eligible_source_years,
    fit_source_matrix,
    map_source_matrix,
    select_consensus,
)


def _rows(pitchers: list[int], contexts: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": pitchers,
            "balls_before": [value[0] for value in contexts],
            "strikes_before": [value[1] for value in contexts],
            "batter_hand": [value[2] for value in contexts],
        }
    )


def test_source_years_are_strictly_prior() -> None:
    assert eligible_source_years(2022) == (2020, 2021)
    assert eligible_source_years(2023) == (2020, 2021, 2022)
    assert eligible_source_years(2024) == (2020, 2021, 2022, 2023)


def test_low_rank_mapping_is_deterministic_and_unseen_is_zero() -> None:
    rows = _rows(
        [10, 10, 20, 20],
        [(0, 0, 1), (0, 1, 1), (0, 0, 1), (0, 1, 1)],
    )
    target = np.array([1.0, 0.0, 0.0, 1.0])
    parent = np.full(4, 0.5)
    model = fit_source_matrix(
        rows,
        target,
        parent,
        smoothing_grid=(1.0,),
        rank_grid=(1,),
    )
    query = _rows([10, 999], [(0, 0, 1), (0, 0, 1)])
    first, seen, exact = map_source_matrix(model, query)
    second, _, _ = map_source_matrix(model, query)
    assert np.array_equal(seen, np.array([True, False]))
    assert np.array_equal(exact, np.array([True, False]))
    assert first["lowrank_s1_r1"][1] == 0.0
    assert np.array_equal(first["lowrank_s1_r1"], second["lowrank_s1_r1"])
    assert abs(model["diagnostics"]["residual_mean_after_centering"]) < 1e-12


def test_bank_averages_unseen_source_as_zero() -> None:
    source_rows = _rows([1, 1], [(0, 0, 1), (0, 1, 1)])
    model_seen = fit_source_matrix(
        source_rows,
        np.array([1.0, 0.0]),
        np.array([0.4, 0.4]),
        smoothing_grid=(1.0,),
        rank_grid=(1,),
    )
    model_unseen = fit_source_matrix(
        _rows([2, 2], [(0, 0, 1), (0, 1, 1)]),
        np.array([1.0, 0.0]),
        np.array([0.4, 0.4]),
        smoothing_grid=(1.0,),
        rank_grid=(1,),
    )
    bank, coverage = build_audit_bank(
        {2020: model_seen, 2021: model_unseen},
        _rows([1], [(0, 0, 1)]),
        2022,
    )
    mapped, _, _ = map_source_matrix(model_seen, _rows([1], [(0, 0, 1)]))
    assert np.allclose(bank["lowrank_s1_r1"], mapped["lowrank_s1_r1"] / 2.0)
    assert coverage["pitcher_seen_any_rate"] == 1.0
    assert coverage["pitcher_seen_every_rate"] == 0.0


def test_consensus_uses_only_shared_recipe_columns() -> None:
    base = {
        "signal": "lowrank_s300_r2",
        "domain": "ALL",
        "weight": 0.2,
        "positive_month_fraction": 1.0,
        "worst_month_gain": 2.0,
        "applied_domain_gain": 1.0,
        "minimum_domain_gain": 1.0,
        "selection_score": 1.0,
        "mean_abs_shift": 0.001,
    }
    stage1 = pd.DataFrame([{**base, "gain": 3.0}])
    stage2 = pd.DataFrame([{**base, "gain": 4.0}])
    merged, chosen = select_consensus(stage1, stage2)
    assert len(merged) == 1
    assert bool(chosen["passes_consensus_gate"])
    assert chosen["minimum_gain"] == 3.0
