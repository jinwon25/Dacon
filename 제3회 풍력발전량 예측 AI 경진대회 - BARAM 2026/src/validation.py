from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_audit import prediction_day
from src.metrics import evaluate_group


@dataclass(frozen=True)
class TemporalFold:
    name: str
    target: str
    train_years: tuple[int, ...]
    valid_year: int
    train: np.ndarray
    valid: np.ndarray

    def assert_causal(self, index: pd.DatetimeIndex) -> None:
        if not self.train.any() or not self.valid.any():
            raise ValueError(f"Fold {self.name} has an empty train or validation partition")
        if index[self.train].max() >= index[self.valid].min():
            raise ValueError(f"Fold {self.name} is not strictly past-to-future")
        if np.any(self.train & self.valid):
            raise ValueError(f"Fold {self.name} has overlapping partitions")


def make_expanding_year_folds(
    index: Iterable[pd.Timestamp],
    labels: pd.Series,
    target: str,
) -> list[TemporalFold]:
    """Create all feasible past-year(s) -> next-year folds.

    On the supplied data this yields 2022->2023 and 2022~2023->2024 for
    groups 1/2, and 2023->2024 for group 3.
    """
    index = pd.DatetimeIndex(index)
    if len(index) != len(labels):
        raise ValueError("index and labels must have equal length")
    available = labels.notna().to_numpy()
    forecast_days = prediction_day(index)
    forecast_year = forecast_days.year
    years = sorted(int(year) for year in np.unique(forecast_year[available]))
    folds: list[TemporalFold] = []
    for valid_year in years[1:]:
        train_years = tuple(year for year in years if year < valid_year)
        train = available & np.isin(forecast_year, train_years)
        valid = available & (forecast_year == valid_year)
        fold = TemporalFold(
            name=f"{'_'.join(map(str, train_years))}_to_{valid_year}",
            target=target,
            train_years=train_years,
            valid_year=valid_year,
            train=train,
            valid=valid,
        )
        fold.assert_causal(index)
        folds.append(fold)
    if not folds:
        raise ValueError(f"Target {target} needs at least two label years for year-forward validation")
    return folds


def assign_oof_roles(index: Iterable[pd.Timestamp]) -> np.ndarray:
    """Separate model selection, calibration, and final evaluation in time.

    The penultimate OOF year is split into H1 selection and H2 calibration;
    the latest OOF year is evaluation-only. If only one OOF year exists, all
    rows are evaluation-only and calibration must fail closed.
    """
    index = pd.DatetimeIndex(index)
    forecast_days = prediction_day(index)
    years = sorted(np.unique(forecast_days.year).tolist())
    roles = np.full(len(index), "unused", dtype=object)
    if not years:
        return roles
    evaluation_year = int(years[-1])
    roles[forecast_days.year == evaluation_year] = "evaluation"
    if len(years) >= 2:
        development_year = int(years[-2])
        development = forecast_days.year == development_year
        roles[development & (forecast_days.month <= 6)] = "selection"
        roles[development & (forecast_days.month >= 7)] = "calibration"
    return roles


def issue_day_block_bootstrap(
    index: Iterable[pd.Timestamp],
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    capacity: float,
    repetitions: int = 2_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Paired confidence interval for candidate-minus-baseline Score by issue day."""
    index = pd.DatetimeIndex(index)
    y_true = np.asarray(y_true, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not (len(index) == len(y_true) == len(baseline) == len(candidate)):
        raise ValueError("bootstrap inputs must have equal length")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    days = prediction_day(index)
    unique_days = days.unique()
    positions = {day: np.flatnonzero(days == day) for day in unique_days}
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        sampled = rng.choice(unique_days, size=len(unique_days), replace=True)
        take = np.concatenate([positions[day] for day in sampled])
        base_metric = evaluate_group(y_true[take], baseline[take], capacity)
        candidate_metric = evaluate_group(y_true[take], candidate[take], capacity)
        deltas[iteration] = candidate_metric.score - base_metric.score
    return {
        "repetitions": repetitions,
        "issue_days": int(len(unique_days)),
        "mean": float(deltas.mean()),
        "q025": float(np.quantile(deltas, 0.025)),
        "q05": float(np.quantile(deltas, 0.05)),
        "q50": float(np.quantile(deltas, 0.50)),
        "q95": float(np.quantile(deltas, 0.95)),
        "q975": float(np.quantile(deltas, 0.975)),
        "positive_fraction": float(np.mean(deltas > 0)),
    }
