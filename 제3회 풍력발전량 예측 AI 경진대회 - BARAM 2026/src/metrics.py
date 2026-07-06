from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


CAPACITY_KWH = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}


@dataclass(frozen=True)
class MetricResult:
    score: float
    one_minus_nmae: float
    ficr: float
    nmae: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
) -> MetricResult:
    """Evaluate one group on rows used by the competition.

    FICR currently follows the public Korean settlement convention:
    error <= 6% earns 4 units, error <= 8% earns 3 units, otherwise 0.
    Replace only this function if DACON's downloadable scorer differs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0.10 * capacity)
    if not valid.any():
        raise ValueError("No valid evaluation rows (actual must be >= 10% of capacity).")

    error_rate = np.abs(y_true[valid] - y_pred[valid]) / capacity
    nmae = float(error_rate.mean())
    settlement = np.where(error_rate <= 0.06, 4.0, np.where(error_rate <= 0.08, 3.0, 0.0))
    ficr = float(settlement.mean() / 4.0)
    one_minus_nmae = 1.0 - nmae
    score = 0.5 * one_minus_nmae + 0.5 * ficr
    return MetricResult(score, one_minus_nmae, ficr, nmae, int(valid.sum()))


def evaluate_competition(
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
) -> dict[str, object]:
    groups = {
        target: evaluate_group(y_true[target], y_pred[target], capacity)
        for target, capacity in CAPACITY_KWH.items()
    }
    mean_nmae = float(np.mean([x.nmae for x in groups.values()]))
    mean_ficr = float(np.mean([x.ficr for x in groups.values()]))
    return {
        "score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
        "one_minus_nmae": 1.0 - mean_nmae,
        "ficr": mean_ficr,
        "groups": {k: v.to_dict() for k, v in groups.items()},
    }
