from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_audit import prediction_day
from src.metrics import evaluate_group


def detailed_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0.10 * capacity)
    if not valid.any():
        return {name: np.nan for name in (
            "score", "one_minus_nmae", "ficr", "nmae", "hit_rate_6pct", "hit_rate_8pct",
            "actual_weighted_hit_rate_6pct", "actual_weighted_hit_rate_8pct", "bias_kwh", "residual_std_kwh",
        )} | {"n_samples": 0}
    actual = y_true[valid]
    pred = y_pred[valid]
    official = evaluate_group(actual, pred, capacity)
    error_rate = np.abs(actual - pred) / capacity
    residual = pred - actual
    actual_sum = actual.sum()
    return official.to_dict() | {
        "hit_rate_6pct": float(np.mean(error_rate <= 0.06)),
        "hit_rate_8pct": float(np.mean(error_rate <= 0.08)),
        "actual_weighted_hit_rate_6pct": float(np.sum(actual * (error_rate <= 0.06)) / actual_sum),
        "actual_weighted_hit_rate_8pct": float(np.sum(actual * (error_rate <= 0.08)) / actual_sum),
        "bias_kwh": float(np.mean(residual)),
        "residual_std_kwh": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
    }


def _evaluate_categories(frame: pd.DataFrame, category: pd.Series, capacity: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in category.dropna().unique():
        mask = category == value
        output[str(value)] = detailed_group_metrics(
            frame.loc[mask, "y_true"].to_numpy(),
            frame.loc[mask, "y_pred"].to_numpy(),
            capacity,
        )
    return output


def slice_diagnostics(frame: pd.DataFrame, capacity: float) -> dict[str, Any]:
    """Evaluate OOF predictions across only inference-available diagnostic axes.

    `actual_power_bin` is diagnostic-only and must never be fed to a model or
    calibrator. Optional wind/disagreement columns should originate from train
    NWP rows only.
    """
    required = {"timestamp", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    result: dict[str, Any] = {"overall": detailed_group_metrics(work["y_true"], work["y_pred"], capacity)}
    result["hour"] = _evaluate_categories(work, work["timestamp"].dt.hour, capacity)
    result["month"] = _evaluate_categories(work, work["timestamp"].dt.to_period("M").astype(str), capacity)
    season = work["timestamp"].dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    result["season"] = _evaluate_categories(work, season, capacity)
    actual_fraction = work["y_true"] / capacity
    actual_bins = pd.cut(actual_fraction, bins=[0.10, 0.20, 0.40, 0.60, 0.80, 1.00, np.inf], right=False, include_lowest=True)
    result["actual_power_bin_diagnostic_only"] = _evaluate_categories(work, actual_bins, capacity)
    predicted_fraction = work["y_pred"] / capacity
    predicted_bins = pd.cut(
        predicted_fraction,
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, np.inf],
        right=False,
    )
    result["predicted_power_bin"] = _evaluate_categories(work, predicted_bins, capacity)
    if "lead_hour" in work:
        lead_bins = pd.cut(work["lead_hour"], bins=[11.5, 17.5, 23.5, 29.5, 35.5], include_lowest=True)
        result["lead_time_bucket"] = _evaluate_categories(work, lead_bins, capacity)
    if "wind_speed" in work:
        wind_bins = pd.cut(work["wind_speed"], bins=[-np.inf, 4, 7, 10, 13, np.inf])
        result["wind_speed_bucket"] = _evaluate_categories(work, wind_bins, capacity)
    if "wind_direction" in work:
        direction = np.mod(work["wind_direction"], 360.0)
        sector = pd.cut(direction, bins=np.arange(0, 361, 45), right=False, include_lowest=True)
        result["wind_direction_sector"] = _evaluate_categories(work, sector, capacity)
    if "nwp_disagreement" in work:
        finite = work["nwp_disagreement"].replace([np.inf, -np.inf], np.nan)
        try:
            disagreement = pd.qcut(finite, q=4, duplicates="drop")
            result["ldaps_gfs_disagreement_quartile"] = _evaluate_categories(work, disagreement, capacity)
        except ValueError:
            result["ldaps_gfs_disagreement_quartile"] = {}
    if "predictive_interval_width" in work:
        finite = work["predictive_interval_width"].replace([np.inf, -np.inf], np.nan)
        try:
            width = pd.qcut(finite, q=4, duplicates="drop")
            result["predictive_interval_width_quartile"] = _evaluate_categories(work, width, capacity)
        except ValueError:
            result["predictive_interval_width_quartile"] = {}

    # Truth-based ramp buckets are diagnostics only.  They are intentionally
    # computed inside each day-ahead issue so adjacent forecast days never mix.
    work = work.sort_values("timestamp").copy()
    work["_forecast_day"] = prediction_day(work["timestamp"])
    work["_actual_ramp"] = work.groupby("_forecast_day", sort=False)["y_true"].diff().abs() / capacity
    try:
        ramp = pd.qcut(work["_actual_ramp"], q=4, duplicates="drop")
        result["actual_ramp_magnitude_quartile_diagnostic_only"] = _evaluate_categories(work, ramp, capacity)
    except ValueError:
        result["actual_ramp_magnitude_quartile_diagnostic_only"] = {}
    return result
