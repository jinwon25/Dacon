from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v194_hierarchical_context_transport import (
    _select,
    apply_local,
    context_labels,
)


def test_apply_local_changes_only_exact_gate_intersection() -> None:
    base = np.array([0.4, 0.5, 0.6])
    local = np.array([0.8, 0.1, 0.2])
    output = apply_local(
        base, local,
        exact=np.array([True, True, False]),
        gate=np.array([True, False, True]),
        weight=0.1,
    )
    np.testing.assert_allclose(output, [0.44, 0.5, 0.6])


def test_context_labels_are_row_aligned() -> None:
    frame = pd.DataFrame(
        {
            "balls_before": [0, 3], "strikes_before": [2, 2],
            "num_runners_on": [0, 2], "li": [0.5, 2.0], "inning": [2, 8],
            "score_diff_pitcher_team": [0, -3], "asof_pitcher_n": [10, 1200],
            "asof_batter_n": [100, 400], "game_month": [4, 9],
            "pitcher_hand": [1, 2], "batter_hand": [1, 1],
            "top_bottom": ["T", "B"], "outs_before": [0, 2],
        }
    )
    labels = context_labels(frame)
    assert all(len(value) == 2 for value in labels.values())
    assert labels["count"].tolist() == ["0-2", "3-2"]
    assert labels["runner"].tolist() == ["bases_empty", "runner_on"]


def test_select_uses_only_requested_fit_axes() -> None:
    rows = pd.DataFrame(
        [
            {
                "gate": "a", "weight": 0.0025, "minimum_active_fraction": 0.5,
                "full_2022_gain": 2.0, "full_2022_month_fraction": 1.0,
                "full_2022_worst_month": 1.0, "late_2023_gain": 2.0,
                "late_2023_month_fraction": 1.0, "late_2023_worst_month": 1.0,
                "full_2024_gain": -100.0, "full_2024_month_fraction": 0.0,
                "full_2024_worst_month": -100.0,
            },
            {
                "gate": "all", "weight": 0.0025, "minimum_active_fraction": 1.0,
                "full_2022_gain": 1.0, "full_2022_month_fraction": 1.0,
                "full_2022_worst_month": 1.0, "late_2023_gain": 1.0,
                "late_2023_month_fraction": 1.0, "late_2023_worst_month": 1.0,
                "full_2024_gain": 100.0, "full_2024_month_fraction": 1.0,
                "full_2024_worst_month": 100.0,
            },
        ]
    )
    selected = _select(rows, ("full_2022", "late_2023"))
    assert selected["gate"] == "a"
