from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.mechanism_diversity_blend_audit import (
    TARGETS,
    _mixtures,
    align_to_index,
    compose_displacement_blend,
    q1_q2_stability_score,
)


def test_align_to_index_fills_missing_rows_from_anchor() -> None:
    target = pd.date_range("2024-01-01", periods=4, freq="h")
    source = target[[0, 2, 3]]

    aligned, count = align_to_index(
        target,
        source,
        np.array([10.0, 30.0, 40.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    assert count == 3
    assert aligned.tolist() == [10.0, 2.0, 30.0, 40.0]


def test_displacement_blend_keeps_group3_frozen() -> None:
    active = {
        target: np.full(3, 100.0 + position)
        for position, target in enumerate(TARGETS)
    }
    mechanisms = {
        "tabular": {
            target: values + (10.0 if target != "kpx_group_3" else 0.0)
            for target, values in active.items()
        },
        "trajectory": {
            target: values + (20.0 if target != "kpx_group_3" else 0.0)
            for target, values in active.items()
        },
    }

    output = compose_displacement_blend(
        active,
        mechanisms,
        {"tabular": 0.25, "trajectory": 0.75},
        total_alpha=0.20,
    )

    assert np.allclose(output["kpx_group_1"], active["kpx_group_1"] + 3.5)
    assert np.allclose(output["kpx_group_2"], active["kpx_group_2"] + 3.5)
    assert np.array_equal(output["kpx_group_3"], active["kpx_group_3"])


def test_mixture_grid_contains_sparse_one_to_three_member_blends() -> None:
    mixtures = _mixtures(("a", "b", "c", "d"))

    assert len(mixtures) == 38
    assert {"a": 1.0} in mixtures
    assert {"a": 0.5, "b": 0.5} in mixtures
    assert {"a": 0.5, "b": 0.25, "c": 0.25} in mixtures
    assert all(np.isclose(sum(record.values()), 1.0) for record in mixtures)


def test_q1_q2_stability_score_uses_only_pre_h2_score_deltas() -> None:
    record = {
        "q1_delta": {"score": 0.4},
        "q1_monthly_deltas": {
            "1": {"score": 0.3},
            "2": {"score": 0.2},
            "3": {"score": 0.1},
        },
        "q2_delta_if_q1_eligible": {"score": 0.25},
        "q2_monthly_deltas_if_q1_eligible": {
            "4": {"score": 0.15},
            "5": {"score": 0.05},
            "6": {"score": 0.12},
        },
        "posthoc_period_deltas": {"h2": {"score": -999.0}},
    }

    assert q1_q2_stability_score(record) == 0.05
