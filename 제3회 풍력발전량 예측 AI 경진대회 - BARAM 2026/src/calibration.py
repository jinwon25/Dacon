from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import evaluate_group


@dataclass
class ShrunkResidualCalibrator:
    capacity: float
    min_samples: int = 200
    power_bins: int = 5
    strength: float = 0.0
    global_offset: float = 0.0
    corrections: dict[tuple[int, int], float] | None = None

    @staticmethod
    def _lead_bucket(lead_hour: np.ndarray) -> np.ndarray:
        return np.clip(((np.asarray(lead_hour, dtype=float) - 12.0) // 6.0).astype(int), 0, 3)

    def _power_bucket(self, prediction: np.ndarray) -> np.ndarray:
        fraction = np.clip(np.asarray(prediction, dtype=float) / self.capacity, 0.0, 0.999999)
        return np.minimum((fraction * self.power_bins).astype(int), self.power_bins - 1)

    def fit(
        self,
        y_true: np.ndarray,
        prediction: np.ndarray,
        lead_hour: np.ndarray,
        strengths: tuple[float, ...] = (0.0, 0.5, 1.0),
    ) -> "ShrunkResidualCalibrator":
        y_true = np.asarray(y_true, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        lead_hour = np.asarray(lead_hour, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(prediction) & np.isfinite(lead_hour) & (y_true >= 0.10 * self.capacity)
        if valid.sum() < self.min_samples:
            raise ValueError("Not enough eligible OOF rows to fit calibration")
        residual = y_true[valid] - prediction[valid]
        self.global_offset = float(np.median(residual))
        power_bucket = self._power_bucket(prediction[valid])
        lead_bucket = self._lead_bucket(lead_hour[valid])
        self.corrections = {}
        for pbin in range(self.power_bins):
            for lbin in range(4):
                mask = (power_bucket == pbin) & (lead_bucket == lbin)
                count = int(mask.sum())
                local = float(np.median(residual[mask])) if count else self.global_offset
                shrink = count / (count + self.min_samples)
                self.corrections[(pbin, lbin)] = shrink * local + (1.0 - shrink) * self.global_offset
        best = (-np.inf, 0.0)
        for strength in strengths:
            self.strength = float(strength)
            calibrated = self.predict(prediction[valid], lead_hour[valid])
            score = evaluate_group(y_true[valid], calibrated, self.capacity).score
            if score > best[0]:
                best = (score, float(strength))
        self.strength = best[1]
        return self

    def predict(self, prediction: np.ndarray, lead_hour: np.ndarray) -> np.ndarray:
        if self.corrections is None:
            raise ValueError("Calibrator is not fitted")
        prediction = np.asarray(prediction, dtype=float)
        power_bucket = self._power_bucket(prediction)
        lead_bucket = self._lead_bucket(np.asarray(lead_hour, dtype=float))
        correction = np.asarray([self.corrections[(int(p), int(l))] for p, l in zip(power_bucket, lead_bucket)])
        return np.clip(prediction + self.strength * correction, 0.0, self.capacity)


def optimize_simplex_blend(
    predictions: np.ndarray,
    y_true: np.ndarray,
    capacity: float,
    trials: int = 5_000,
    seed: int = 2026,
) -> tuple[np.ndarray, dict[str, float | int]]:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 1:
        raise ValueError("predictions must have shape [rows, models]")
    if len(y_true) != len(predictions):
        raise ValueError("predictions and y_true must have equal rows")
    rng = np.random.default_rng(seed)
    candidates = np.vstack([np.eye(predictions.shape[1]), np.full((1, predictions.shape[1]), 1 / predictions.shape[1]), rng.dirichlet(np.ones(predictions.shape[1]), size=trials)])
    best_score = -np.inf
    best_weight = candidates[0]
    best_metric = None
    for weight in candidates:
        pred = np.clip(predictions @ weight, 0.0, capacity)
        metric = evaluate_group(y_true, pred, capacity)
        if metric.score > best_score:
            best_score = metric.score
            best_weight = weight.copy()
            best_metric = metric.to_dict()
    assert best_metric is not None
    return best_weight, best_metric
