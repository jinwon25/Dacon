import numpy as np
import pandas as pd
import pytest

from src.archive.v32_temporal_consensus_eb import _split_month, consensus_effect


def _frame(residual):
    return pd.DataFrame(
        {
            "game_month": [3, 4, 7, 8],
            "pitcher_id": [1, 1, 1, 1],
            "batter_hand": [2, 2, 2, 2],
            "residual_v19": residual,
        }
    )


def test_split_month_creates_two_nonempty_chronological_halves():
    assert _split_month(_frame([0.1] * 4)) == 4


def test_consensus_keeps_same_sign_and_rejects_reversal():
    audit = _frame([0.0] * 4).iloc[:1]
    positive, active_positive = consensus_effect(
        _frame([0.2, 0.1, 0.3, 0.1]),
        audit,
        ("pitcher_id", "batter_hand"),
        alpha=1.0,
        min_half_count=1.0,
        mode="full",
    )
    reversed_effect, active_reversed = consensus_effect(
        _frame([0.2, 0.1, -0.3, -0.1]),
        audit,
        ("pitcher_id", "batter_hand"),
        alpha=1.0,
        min_half_count=1.0,
        mode="full",
    )
    assert positive[0] > 0.0
    assert active_positive[0]
    assert reversed_effect[0] == pytest.approx(0.0)
    assert not active_reversed[0]


def test_consensus_requires_support_in_both_halves():
    fit = _frame([0.2, 0.1, 0.3, 0.1])
    audit = fit.iloc[:1]
    effect, active = consensus_effect(
        fit,
        audit,
        ("pitcher_id", "batter_hand"),
        alpha=1.0,
        min_half_count=3.0,
        mode="minimum",
    )
    assert effect[0] == pytest.approx(0.0)
    assert not active[0]
