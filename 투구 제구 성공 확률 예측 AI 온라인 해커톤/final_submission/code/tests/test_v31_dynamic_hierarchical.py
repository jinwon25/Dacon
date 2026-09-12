import numpy as np
import pandas as pd
import pytest

from src.archive.v31_dynamic_hierarchical_residual import (
    _apply,
    _prepare,
    hierarchical_effect,
)


def _fit() -> pd.DataFrame:
    return _prepare(
        pd.DataFrame(
            {
                "pitcher_id": [1, 1, 1, 2],
                "domain3": ["R_CORE"] * 4,
                "batter_hand": [1, 1, 2, 1],
                "balls_before": [0, 0, 1, 0],
                "strikes_before": [0, 0, 1, 0],
                "game_month": [6, 7, 7, 7],
                "residual_v19": [0.2, 0.2, -0.1, -0.2],
            }
        )
    )


def test_hierarchical_effect_pools_fine_groups_toward_player_parent():
    fit = _fit()
    audit = _prepare(
        pd.DataFrame(
            {
                "pitcher_id": [1, 1, 3],
                "domain3": ["R_CORE"] * 3,
                "batter_hand": [1, 2, 1],
                "balls_before": [0, 2, 0],
                "strikes_before": [0, 2, 0],
            }
        )
    )
    levels = (
        ("pitcher_id",),
        ("pitcher_id", "domain3"),
        ("pitcher_id", "domain3", "batter_hand"),
        ("pitcher_id", "domain3", "batter_hand", "count_state"),
    )
    result = hierarchical_effect(fit, audit, levels, (10.0, 10.0, 10.0, 10.0), None)
    assert result[0] > 0.0
    assert result[1] > 0.0  # unseen fine count falls back to a known hand/player parent
    assert result[2] == pytest.approx(0.0)  # unseen player falls back to zero


def test_recency_weighting_changes_the_estimate():
    fit = _fit()
    audit = fit.iloc[[0]].copy()
    levels = (("pitcher_id",),)
    uniform = hierarchical_effect(fit, audit, levels, (1.0,), None)
    recent = hierarchical_effect(fit, audit, levels, (1.0,), 0.5)
    assert recent[0] != pytest.approx(uniform[0])


def test_prepare_reconstructs_three_level_pressure():
    prepared = _prepare(
        pd.DataFrame(
            {
                "balls_before": [3, 1, 1],
                "strikes_before": [0, 2, 1],
            }
        )
    )
    assert prepared["pressure"].tolist() == ["threeball", "twostrike", "normal"]


def test_apply_changes_only_declared_domain():
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.5],
            "v25": [0.4, 0.5],
            "domain3": ["R_CORE", "F"],
        }
    )
    candidate, mask = _apply(frame, np.array([0.1, 0.1]), 0.2, "R_CORE")
    assert mask.tolist() == [True, False]
    assert candidate == pytest.approx([0.42, 0.5])
