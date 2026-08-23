from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v57_public_strict_blend import (
    _axis_diversity,
    _mode_from_signal,
    blend_candidate,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [1.0, 0.0, 1.0, 0.0],
            "game_month": [8, 8, 9, 9],
            "domain3": ["R_CORE", "R_ANCHOR", "F", "R_CORE"],
        }
    )


def test_probability_blend_changes_only_declared_route() -> None:
    frame = _frame()
    parent = np.array([0.6, 0.4, 0.7, 0.3])
    strict = np.array([0.8, 0.2, 0.5, 0.5])
    candidate, active = blend_candidate(
        frame,
        parent,
        strict,
        mode="probability",
        route="R_CORE",
        weight=0.25,
    )
    assert active.tolist() == [True, False, False, True]
    assert candidate.tolist() == pytest.approx([0.65, 0.4, 0.7, 0.35])


def test_logit_blend_is_convex_in_log_odds_and_row_local() -> None:
    frame = _frame()
    parent = np.array([0.2, 0.4, 0.7, 0.8])
    strict = np.array([0.8, 0.6, 0.3, 0.2])
    candidate, active = blend_candidate(
        frame,
        parent,
        strict,
        mode="logit",
        route="ALL",
        weight=0.5,
    )
    assert active.all()
    assert candidate.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_strict_blend_rejects_invalid_recipe() -> None:
    frame = _frame()
    value = np.full(len(frame), 0.5)
    with pytest.raises(ValueError):
        blend_candidate(
            frame, value, value, mode="future", route="ALL", weight=0.1
        )
    with pytest.raises(ValueError):
        blend_candidate(
            frame, value, value, mode="probability", route="R_UNKNOWN", weight=0.1
        )
    with pytest.raises(ValueError):
        blend_candidate(
            frame, value, value, mode="probability", route="ALL", weight=1.1
        )


def test_axis_diversity_recovers_optimal_linear_weight() -> None:
    frame = _frame()
    parent = np.array([0.4, 0.4, 0.6, 0.6])
    strict = np.array([0.6, 0.2, 0.8, 0.4])
    result = _axis_diversity(frame, parent, strict)
    assert result["same_axis_unconstrained_probability_weight"] == pytest.approx(2.5)
    assert result["same_axis_convex_probability_weight"] == pytest.approx(1.0)


def test_mode_signal_round_trip() -> None:
    assert _mode_from_signal("strict_probability") == "probability"
    assert _mode_from_signal("strict_logit") == "logit"
    with pytest.raises(ValueError):
        _mode_from_signal("strict_future")
