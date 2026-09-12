from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v236_public1175_pitcher_context_residual import (
    apply_direction,
    context_values,
    fit_differential,
    restrictions,
)


def test_context_values_and_centered_pitcher_contrast() -> None:
    rows = pd.DataFrame({
        "pitcher_hand": ["R", "R", "L", "L"],
        "batter_hand": ["R", "L", "L", "R"],
        "strikes_before": [2, 1, 2, 0],
        "num_runners_on": [1, 0, 0, 2],
    })
    values = context_values(rows)
    assert values["same_hand"].tolist() == [1, 0, 1, 0]
    assert values["two_strike"].tolist() == [1, 0, 1, 0]
    assert values["runner_on"].tolist() == [1, 0, 0, 1]
    table = fit_differential(
        np.array([7, 7, 7, 7]), np.array([0, 0, 1, 1]),
        np.array([-0.1, -0.1, 0.1, 0.1]), shrink=0.0,
    )
    assert np.isclose(table.loc[7], 0.2)


def test_apply_direction_changes_only_frozen_active_support() -> None:
    base = np.array([0.5, 0.6, 0.7])
    direction = np.array([0.1, -0.1, 0.1])
    actual = apply_direction(base, direction, np.array([True, False, True]), 0.5)
    np.testing.assert_allclose(actual, [0.55, 0.6, 0.75])
    audit = restrictions()
    assert audit["public1175_formula_frozen"]
    assert audit["fallback_support_frozen"]
    assert audit["strict_forward_residual_tables"]
    assert not audit["external_code_or_weights_used"]
    assert not audit["test_csv_read"]
