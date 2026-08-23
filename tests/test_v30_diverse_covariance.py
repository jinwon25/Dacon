import numpy as np
import pandas as pd
import pytest

from src.core.diagnostics import compose, diagnostics, v27_parent
from src.core.axes import _quadratic_gain


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0.0, 1.0, 0.0, 1.0],
            "v21": [0.4, 0.6, 0.4, 0.6],
            "v22": [0.41, 0.59, 0.42, 0.58],
            "v25": [0.425, 0.575, 0.42, 0.58],
            "domain3": ["R_ANCHOR", "R_ANCHOR", "R_CORE", "F"],
            "game_month": [8, 8, 9, 9],
        }
    )


def test_v27_parent_scales_only_existing_v25_direction():
    frame = _frame()
    value = v27_parent(frame)
    assert value == pytest.approx([0.43, 0.57, 0.42, 0.58])


def test_compose_respects_mask_and_supports_two_prediction_modes():
    parent = np.array([0.4, 0.6, 0.5])
    v21 = np.array([0.3, 0.7, 0.5])
    signal = np.array([0.5, 0.5, 0.9])
    mask = np.array([True, False, True])
    toward = compose(
        parent, v21, signal, mask, kind="prediction", mode="toward_parent", weight=0.5
    )
    delta = compose(
        parent, v21, signal, mask, kind="prediction", mode="delta_v21", weight=0.5
    )
    assert toward == pytest.approx([0.45, 0.6, 0.7])
    assert delta == pytest.approx([0.5, 0.6, 0.7])


def test_compose_rejects_unknown_mode():
    with pytest.raises(ValueError):
        compose(
            np.array([0.5]),
            np.array([0.5]),
            np.array([0.5]),
            np.array([True]),
            kind="prediction",
            mode="unknown",
            weight=0.1,
        )


def test_diagnostics_reports_month_and_domain_stability():
    frame = _frame()
    parent = v27_parent(frame)
    candidate = parent.copy()
    candidate[0] -= 0.01
    candidate[1] += 0.01
    result = diagnostics(frame, parent, candidate, np.array([True, True, False, False]))
    assert result["gain"] > 0.0
    assert result["positive_month_fraction"] == 1.0
    assert set(result["domain_gains"]) == {"R_CORE", "R_ANCHOR", "F"}


def test_quadratic_gain_matches_direct_brier_difference():
    target = np.array([0.0, 1.0, 1.0, 0.0])
    parent = np.array([0.4, 0.6, 0.55, 0.45])
    direction = np.array([-0.02, 0.03, 0.01, -0.01])
    weights = np.array([0.1, 0.5])
    fast = _quadratic_gain(
        target, parent, direction, np.ones(len(target), dtype=bool), weights
    )
    reference = target.mean() * (1.0 - target.mean())
    direct = []
    for weight in weights:
        candidate = parent + weight * direction
        direct.append(
            100_000.0
            * (
                np.mean(np.square(target - parent))
                - np.mean(np.square(target - candidate))
            )
            / reference
        )
    assert fast == pytest.approx(direct)
