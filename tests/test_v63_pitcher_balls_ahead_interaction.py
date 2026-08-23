from __future__ import annotations

import numpy as np
import pandas as pd

from src.v63_pitcher_balls_ahead_interaction import (
    build_audit_bank,
    eligible_source_years,
    fit_source_interaction,
    map_source_interaction,
)


def _rows(pitchers: list[int], counts: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": pitchers,
            "balls_before": [value[0] for value in counts],
            "strikes_before": [value[1] for value in counts],
        }
    )


def test_history_windows_are_strictly_pre_origin() -> None:
    assert eligible_source_years(2022, "all") == (2020, 2021)
    assert eligible_source_years(2023, "recent2") == (2021, 2022)
    assert eligible_source_years(2024, "latest1") == (2023,)


def test_interaction_removes_pitcher_main_effect() -> None:
    rows = _rows([1, 1, 1, 1], [(0, 0), (0, 1), (2, 1), (3, 2)])
    target = np.array([0.0, 0.0, 1.0, 1.0])
    parent = np.full(4, 0.5)
    model = fit_source_interaction(
        rows, target, parent, smoothing_grid=(1.0,)
    )
    effect = model["effects"][1.0]
    counts = model["counts"]
    assert effect[0, 0] < 0.0 < effect[0, 1]
    assert abs(float(np.sum(effect[0] * counts[0]))) < 1e-12
    assert model["diagnostics"]["count_weighted_main_effect_max_abs"]["1"] < 1e-12


def test_unseen_and_one_state_pitchers_map_to_zero() -> None:
    source = _rows([1, 1, 2, 2], [(0, 1), (2, 1), (2, 1), (3, 1)])
    model = fit_source_interaction(
        source,
        np.array([0.0, 1.0, 1.0, 1.0]),
        np.full(4, 0.5),
        smoothing_grid=(1.0,),
    )
    query = _rows([1, 2, 999], [(2, 1), (2, 1), (2, 1)])
    mapped, seen, both = map_source_interaction(model, query)
    assert np.array_equal(seen, np.array([True, True, False]))
    assert np.array_equal(both, np.array([True, False, False]))
    assert mapped["ahead_contrast_s1"][1] == 0.0
    assert mapped["ahead_contrast_s1"][2] == 0.0


def test_bank_history_modes_never_use_future_model() -> None:
    rows = _rows([1, 1], [(0, 1), (2, 1)])
    models = {
        year: fit_source_interaction(
            rows,
            np.array([0.0, 1.0]),
            np.full(2, 0.5),
            smoothing_grid=(1.0,),
        )
        for year in (2020, 2021, 2022, 2023)
    }
    bank, coverage = build_audit_bank(models, rows, 2022)
    assert sorted(bank) == [
        "all__ahead_contrast_s1",
        "latest1__ahead_contrast_s1",
        "recent2__ahead_contrast_s1",
    ]
    assert coverage["all"]["source_years"] == [2020, 2021]
    assert coverage["latest1"]["source_years"] == [2021]
