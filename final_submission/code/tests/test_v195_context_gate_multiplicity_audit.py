from __future__ import annotations

import numpy as np

from src.archive.v195_context_gate_multiplicity_audit import _gain, _select_gate


def test_gain_is_positive_for_a_better_candidate() -> None:
    target = np.array([0.0, 1.0, 0.0, 1.0])
    base = np.full(4, 0.5)
    candidate = np.array([0.4, 0.6, 0.4, 0.6])
    assert _gain(target, base, candidate, np.ones(4, dtype=bool)) > 0.0


def test_select_gate_respects_excluded_rows() -> None:
    axes = {
        name: {
            "target": np.array([0.0, 1.0, 0.0, 1.0]),
            "exact_mask": np.ones(4, dtype=bool),
        }
        for name in ("full_2022", "late_2023", "full_2024")
    }
    base = {name: np.full(4, 0.5) for name in axes}
    candidates = {
        "first": {name: np.array([0.4, 0.6, 0.9, 0.1]) for name in axes},
        "second": {name: np.array([0.9, 0.1, 0.4, 0.6]) for name in axes},
    }
    excluded_tail = {name: np.array([False, False, True, True]) for name in axes}
    excluded_head = {name: ~excluded_tail[name] for name in axes}
    assert _select_gate(candidates, axes, base, excluded_tail) == "first"
    assert _select_gate(candidates, axes, base, excluded_head) == "second"
