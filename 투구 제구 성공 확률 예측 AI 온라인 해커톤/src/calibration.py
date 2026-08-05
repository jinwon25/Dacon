"""Calibration, shrinkage, and leakage-safe fixed-weight blend utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import brier_score, validate_probabilities


def _logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(prob, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def apply_logit_offset(probability: np.ndarray, offset: float) -> np.ndarray:
    probability = validate_probabilities(probability)
    return _sigmoid(_logit(probability) + float(offset))


def fit_season_logit_offset(
    seasons: np.ndarray,
    target: np.ndarray,
    forecast_season: int,
    window: int,
) -> tuple[float, float, dict[int, float]]:
    """Extrapolate annual base-rate logits using training labels only.

    The returned offset moves a model anchored at the latest observed season's
    base rate to the forecast season's rate. It never inspects validation/test
    predictions or their batch mean.
    """
    frame = pd.DataFrame({"season": seasons, "target": target})
    rates = frame.groupby("season", sort=True)["target"].mean()
    if len(rates) < window:
        raise ValueError(f"need {window} seasons, got {len(rates)}")
    recent = rates.iloc[-window:]
    x = recent.index.to_numpy(dtype=float)
    y = _logit(recent.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    forecast_logit = float(slope * forecast_season + intercept)
    latest_logit = float(_logit(np.array([rates.iloc[-1]], dtype=float))[0])
    forecast_rate = float(_sigmoid(np.array([forecast_logit]))[0])
    return forecast_logit - latest_logit, forecast_rate, {
        int(year): float(rate) for year, rate in rates.items()
    }


@dataclass
class PlattCalibrator:
    coefficient: float = 1.0
    intercept: float = 0.0

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "PlattCalibrator":
        probability = validate_probabilities(probability)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(_logit(probability).reshape(-1, 1), np.asarray(target))
        self.coefficient = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = validate_probabilities(probability)
        return _sigmoid(self.coefficient * _logit(probability) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt", "coefficient": self.coefficient, "intercept": self.intercept}


@dataclass
class IsotonicCalibrator:
    x_thresholds: list[float] | None = None
    y_thresholds: list[float] | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "IsotonicCalibrator":
        probability = validate_probabilities(probability)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(probability, np.asarray(target))
        self.x_thresholds = model.X_thresholds_.astype(float).tolist()
        self.y_thresholds = model.y_thresholds_.astype(float).tolist()
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = validate_probabilities(probability)
        if self.x_thresholds is None or self.y_thresholds is None:
            raise RuntimeError("calibrator is not fit")
        return np.interp(probability, self.x_thresholds, self.y_thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "isotonic", "x_thresholds": self.x_thresholds, "y_thresholds": self.y_thresholds}


def calibrator_from_dict(spec: dict[str, Any]):
    if spec["method"] == "platt":
        return PlattCalibrator(float(spec["coefficient"]), float(spec["intercept"]))
    if spec["method"] == "isotonic":
        return IsotonicCalibrator(list(spec["x_thresholds"]), list(spec["y_thresholds"]))
    if spec["method"] == "identity":
        return None
    raise ValueError(f"unknown calibration method: {spec['method']}")


def choose_base_rate_shrinkage(
    probability: np.ndarray,
    target: np.ndarray,
    base_rate: float,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.0, 0.20, 41)
    best = (float("inf"), 0.0)
    for weight in grid:
        pred = (1.0 - weight) * probability + weight * base_rate
        score = brier_score(target, pred)
        if score < best[0]:
            best = (score, float(weight))
    return best[1], best[0]


def choose_three_way_blend(
    predictions: list[np.ndarray],
    target: np.ndarray,
    step: float = 0.05,
) -> tuple[list[float], float]:
    if len(predictions) != 3:
        raise ValueError("exactly three prediction vectors are required")
    best_score = float("inf")
    best_weights = [1.0, 0.0, 0.0]
    units = int(round(1.0 / step))
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            weights = np.array([first, second, third], dtype=float) / units
            pred = sum(weight * values for weight, values in zip(weights, predictions))
            score = brier_score(target, pred)
            if score < best_score:
                best_score = score
                best_weights = weights.tolist()
    return best_weights, best_score
