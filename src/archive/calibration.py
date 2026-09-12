"""Calibration, shrinkage, and leakage-safe fixed-weight blend utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
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
class InterceptCalibrator:
    intercept: float = 0.0

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "InterceptCalibrator":
        probability = validate_probabilities(probability)
        target = np.asarray(target, dtype=np.float64)
        logit = _logit(probability)

        def objective(offset: float) -> float:
            pred = np.clip(_sigmoid(logit + offset), 1e-12, 1.0 - 1e-12)
            return float(-np.mean(target * np.log(pred) + (1.0 - target) * np.log1p(-pred)))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"intercept calibration failed: {result.message}")
        self.intercept = float(result.x)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = validate_probabilities(probability)
        return _sigmoid(_logit(probability) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "intercept", "intercept": self.intercept}


@dataclass
class BetaCalibrator:
    coefficient_log_p: float = 1.0
    coefficient_log_one_minus_p: float = 1.0
    intercept: float = 0.0
    regularization: float = 1e-3

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "BetaCalibrator":
        probability = validate_probabilities(probability)
        target = np.asarray(target, dtype=np.float64)
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        log_p = np.log(clipped)
        neg_log_one_minus_p = -np.log1p(-clipped)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            a, b, c = parameters
            value = a * log_p + b * neg_log_one_minus_p + c
            pred = np.clip(_sigmoid(value), 1e-12, 1.0 - 1e-12)
            loss = -np.mean(target * np.log(pred) + (1.0 - target) * np.log1p(-pred))
            penalty = self.regularization * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + c**2)
            residual = pred - target
            gradient = np.array(
                [
                    np.mean(residual * log_p) + 2.0 * self.regularization * (a - 1.0),
                    np.mean(residual * neg_log_one_minus_p)
                    + 2.0 * self.regularization * (b - 1.0),
                    np.mean(residual) + 2.0 * self.regularization * c,
                ]
            )
            return float(loss + penalty), gradient

        result = minimize(
            objective,
            x0=np.array([1.0, 1.0, 0.0]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 5.0), (0.0, 5.0), (-3.0, 3.0)],
        )
        if not result.success:
            raise RuntimeError(f"beta calibration failed: {result.message}")
        self.coefficient_log_p = float(result.x[0])
        self.coefficient_log_one_minus_p = float(result.x[1])
        self.intercept = float(result.x[2])
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = validate_probabilities(probability)
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        value = (
            self.coefficient_log_p * np.log(clipped)
            - self.coefficient_log_one_minus_p * np.log1p(-clipped)
            + self.intercept
        )
        return _sigmoid(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "beta",
            "coefficient_log_p": self.coefficient_log_p,
            "coefficient_log_one_minus_p": self.coefficient_log_one_minus_p,
            "intercept": self.intercept,
            "regularization": self.regularization,
        }


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
    if spec["method"] == "intercept":
        return InterceptCalibrator(float(spec["intercept"]))
    if spec["method"] == "beta":
        return BetaCalibrator(
            coefficient_log_p=float(spec["coefficient_log_p"]),
            coefficient_log_one_minus_p=float(spec["coefficient_log_one_minus_p"]),
            intercept=float(spec["intercept"]),
            regularization=float(spec.get("regularization", 1e-3)),
        )
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


def fit_constrained_blend(
    predictions: list[np.ndarray],
    target: np.ndarray,
    *,
    l2: float = 0.0,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Minimize Brier with nonnegative, sum-one weights."""
    if len(predictions) < 2:
        raise ValueError("at least two prediction vectors are required")
    matrix = np.column_stack([validate_probabilities(values) for values in predictions])
    target = np.asarray(target, dtype=np.float64)
    if matrix.shape[0] != len(target):
        raise ValueError("prediction and target lengths differ")
    n_models = matrix.shape[1]
    if anchor is None:
        anchor = np.full(n_models, 1.0 / n_models)
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (n_models,) or np.any(anchor < 0.0):
        raise ValueError("invalid blend anchor")
    anchor = anchor / anchor.sum()

    def objective(weight: np.ndarray) -> tuple[float, np.ndarray]:
        residual = matrix @ weight - target
        score = np.mean(np.square(residual)) + l2 * np.sum(np.square(weight - anchor))
        gradient = 2.0 * (matrix.T @ residual) / len(target)
        gradient += 2.0 * l2 * (weight - anchor)
        return float(score), gradient

    result = minimize(
        objective,
        x0=anchor,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"blend optimization failed: {result.message}")
    weight = np.clip(result.x, 0.0, 1.0)
    weight /= weight.sum()
    return weight, brier_score(target, matrix @ weight)


def forecast_base_rate(
    season_rates: pd.Series,
    forecast_season: int,
    method: str,
) -> float:
    """One-step season-rate forecast using historical rates only."""
    rates = season_rates.sort_index().astype(float)
    if len(rates) == 0 or int(rates.index.max()) >= forecast_season:
        raise ValueError("season rates must be non-empty and strictly before forecast")
    if method == "last":
        return float(rates.iloc[-1])
    if method.startswith("mean_"):
        window = int(method.split("_", 1)[1])
        if len(rates) < window:
            raise ValueError(f"{method} needs {window} seasons")
        return float(rates.iloc[-window:].mean())
    if method.startswith("logit_trend_"):
        window = int(method.rsplit("_", 1)[1])
        if len(rates) < window:
            raise ValueError(f"{method} needs {window} seasons")
        recent = rates.iloc[-window:]
        slope, intercept = np.polyfit(
            recent.index.to_numpy(dtype=float),
            _logit(recent.to_numpy(dtype=float)),
            deg=1,
        )
        return float(_sigmoid(np.array([slope * forecast_season + intercept]))[0])
    if method.startswith("damped_"):
        _, window_text, phi_text = method.split("_")
        window = int(window_text)
        phi = float(phi_text)
        if len(rates) < window:
            raise ValueError(f"{method} needs {window} seasons")
        recent = rates.iloc[-window:]
        slope, _ = np.polyfit(
            recent.index.to_numpy(dtype=float),
            _logit(recent.to_numpy(dtype=float)),
            deg=1,
        )
        latest_logit = float(_logit(np.array([recent.iloc[-1]]))[0])
        return float(_sigmoid(np.array([latest_logit + phi * slope]))[0])
    raise ValueError(f"unknown base-rate forecast method: {method}")
