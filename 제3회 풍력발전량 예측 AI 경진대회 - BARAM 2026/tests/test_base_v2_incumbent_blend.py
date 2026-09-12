from __future__ import annotations

import numpy as np

from experiments.base_v2_incumbent_blend import choose_weight
from src.metrics import CAPACITY_KWH


def _predictions(offset: float) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    truth: dict[str, np.ndarray] = {}
    prediction: dict[str, np.ndarray] = {}
    for group, capacity in CAPACITY_KWH.items():
        truth[group] = np.full(48, 0.5 * capacity)
        prediction[group] = truth[group] + offset
    return truth, prediction


def test_choose_weight_selects_helpful_member() -> None:
    truth, incumbent = _predictions(600.0)
    _, member = _predictions(0.0)
    weight, records = choose_weight(
        truth,
        incumbent,
        member,
        np.array([0.0, 0.5, 1.0]),
    )
    assert weight == 1.0
    assert len(records) == 3


def test_choose_weight_keeps_zero_for_harmful_member() -> None:
    truth, incumbent = _predictions(0.0)
    _, member = _predictions(600.0)
    weight, _ = choose_weight(
        truth,
        incumbent,
        member,
        np.array([0.0, 0.5, 1.0]),
    )
    assert weight == 0.0
