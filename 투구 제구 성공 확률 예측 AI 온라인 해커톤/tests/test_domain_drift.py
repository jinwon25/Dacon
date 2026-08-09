import numpy as np
import pandas as pd
import pytest

from src.domain_drift import (
    apply_game_type_offsets,
    fit_game_type_regime_offsets,
)


def _history() -> pd.DataFrame:
    rows = []
    rates = {
        2019: {"F": 0.70, "R": 0.50},
        2020: {"F": 0.68, "R": 0.50},
        2021: {"F": 0.72, "R": 0.50},
        2022: {"F": 0.71, "R": 0.50},
        2023: {"F": 0.47, "R": 0.50},
        2024: {"F": 0.46, "R": 0.50},
    }
    for season, groups in rates.items():
        for game_type, rate in groups.items():
            size = 10 if game_type == "F" else 100
            target = np.array(
                [1] * int(rate * size) + [0] * (size - int(rate * size))
            )
            rows.extend(
                {
                    "season": season,
                    "game_type": game_type,
                    "control_success": int(value),
                }
                for value in target
            )
    return pd.DataFrame(rows)


def test_regime_offset_uses_history_and_preserves_unshifted_group():
    offsets, diagnostics = fit_game_type_regime_offsets(
        _history(), 2025, min_group_rows=1
    )
    assert offsets["F"] < 0.0
    assert offsets["R"] == 0.0
    assert diagnostics.loc[diagnostics["game_type"] == "F", "detected"].item()


def test_regime_fit_rejects_forecast_year_target_access():
    with pytest.raises(ValueError):
        fit_game_type_regime_offsets(_history(), 2024, min_group_rows=1)


def test_game_type_offset_is_row_local_and_order_invariant():
    probability = np.array([0.6, 0.6, 0.4])
    game_type = pd.Series(["F", "R", "F"])
    offsets = {"F": -0.1, "R": 0.0}
    together = apply_game_type_offsets(probability, game_type, offsets)
    alone = np.array(
        [
            apply_game_type_offsets(
                probability[[index]], game_type.iloc[[index]], offsets
            )[0]
            for index in range(len(probability))
        ]
    )
    assert np.allclose(together, alone)
    order = np.array([2, 0, 1])
    shuffled = apply_game_type_offsets(
        probability[order], game_type.iloc[order], offsets
    )
    assert np.allclose(shuffled, together[order])
    assert together[1] == probability[1]
