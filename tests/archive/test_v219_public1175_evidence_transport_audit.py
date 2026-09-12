from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v219_public1175_evidence_transport_audit import (
    evaluator_bss_gain,
    pressure_gate,
    restrictions,
    select_source_scale,
    transport,
)


def test_pressure_gate_is_exact_public1175_scope() -> None:
    frame = pd.DataFrame({
        "game_type": ["R", "R", "R", "F"],
        "pitcher_team_id": [1, 13, 1, 1],
        "batter_team_id": [2, 2, 2, 2],
        "num_runners_on": [1, 1, 0, 1],
        "li": [0.5, 2.0, 1.5, 2.0],
    })
    assert pressure_gate(frame).tolist() == [True, False, True, False]


def test_transport_is_paired_and_clipped() -> None:
    result = transport(np.array([0.2, 0.99]), np.array([0.1, 0.1]), 0.5)
    np.testing.assert_allclose(result, [0.25, 0.999])


def test_evaluator_gain_scores_all_rows() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    base = np.full(4, 0.5)
    candidate = np.array([0.4, 0.6, 0.6, 0.4])
    assert evaluator_bss_gain(y, base, candidate) > 0.0


def test_scale_selection_uses_source_consistency() -> None:
    def item(gain: float, fraction: float = 1.0, worst: float = 0.0):
        return {
            "gain": gain,
            "positive_month_fraction": fraction,
            "worst_month_gain": worst,
        }

    results = {
        "0.50": {"full_2022": item(2.0), "late_2023": item(1.0)},
        "1.00": {"full_2022": item(4.0), "late_2023": item(0.5)},
        "1.50": {"full_2022": item(5.0), "late_2023": item(-0.1)},
    }
    assert select_source_scale(results) == 0.5


def test_v219_locks_2024_and_does_not_read_test() -> None:
    audit = restrictions()
    assert audit["preserved_evaluator_parent_axis"]
    assert audit["source_only_scale_selection"]
    assert audit["locked_2024_not_used_for_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
