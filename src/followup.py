"""Multi-year follow-up experiments with strict temporal nesting."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import (
    BetaCalibrator,
    InterceptCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    _logit,
    apply_logit_offset,
    fit_season_logit_offset,
    forecast_base_rate,
)
from src.data import ID_COL, TARGET_COL, read_main
from src.experiment_tracking import append_experiment
from src.metrics import (
    brier_decomposition,
    brier_score,
    brier_skill_score,
    calibration_intercept_slope,
    cluster_bootstrap_delta,
    reliability_table,
)
from src.train import train_lgb_holdout, train_rf_holdout
from src.validation import walk_forward_splits


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _upsert_csv(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8")
        combined = pd.concat([old, frame], ignore_index=True)
        combined = combined.drop_duplicates(keys, keep="last")
    else:
        combined = frame
    _write_csv(path, combined)


def _append_experiment_once(path: Path, record: dict[str, Any]) -> None:
    experiment_id = str(record["experiment_id"])
    if path.exists():
        prior = pd.read_csv(path, usecols=["experiment_id"], encoding="utf-8")
        if experiment_id in set(prior["experiment_id"].astype(str)):
            return
    append_experiment(path, record)


def _season_rates(train: pd.DataFrame, indices: np.ndarray) -> pd.Series:
    return (
        train.iloc[indices]
        .groupby("season", observed=True)[TARGET_COL]
        .mean()
        .sort_index()
        .astype(float)
    )


def _fold_cache_path(project_dir: Path, validation_season: int) -> Path:
    return (
        project_dir
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{validation_season}.npz"
    )


def _load_fold_cache(path: Path, expected_idx: np.ndarray, expected_target: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        result = {name: saved[name] for name in saved.files}
    if not np.array_equal(result["valid_idx"], expected_idx):
        raise ValueError(f"cached validation indices do not match: {path}")
    if not np.array_equal(result["target"], expected_target):
        raise ValueError(f"cached validation targets do not match: {path}")
    return result


def _metric_row(
    experiment_id: str,
    validation_season: int,
    model: str,
    prediction: np.ndarray,
    target: np.ndarray,
    incumbent_prediction: np.ndarray,
    *,
    diagnostic: bool,
    prediction_source: str,
    fit_seasons: str = "",
) -> dict[str, Any]:
    intercept, slope = calibration_intercept_slope(target, prediction)
    return {
        "experiment_id": experiment_id,
        "outer_validation_season": validation_season,
        "validation_role": "diagnostic" if diagnostic else "primary",
        "model": model,
        "n_rows": len(target),
        "brier": brier_score(target, prediction),
        "brier_skill": brier_skill_score(target, prediction),
        "delta_brier": brier_score(target, prediction)
        - brier_score(target, incumbent_prediction),
        "prediction_mean": float(np.mean(prediction)),
        "target_rate": float(np.mean(target)),
        "mean_gap": float(np.mean(prediction) - np.mean(target)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "prediction_source": prediction_source,
        "fit_seasons": fit_seasons,
    }


def _generate_wave0_fold(
    project_dir: Path,
    train: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    validation_season: int,
    config: dict[str, Any],
    official_features: list[str],
) -> dict[str, Any]:
    cache_path = _fold_cache_path(project_dir, validation_season)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    target = train.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.int8)
    if cache_path.exists():
        print(f"[Wave 0] Loading cache for {validation_season}: {cache_path}")
        return _load_fold_cache(cache_path, valid_idx, target)

    incumbent = config["incumbent"]
    print(f"[Wave 0] Training LightGBM for validation {validation_season}...")
    lgb_result = train_lgb_holdout(
        train,
        train_idx,
        valid_idx,
        incumbent["lgb_variant"],
        int(incumbent["max_boost_rounds"]),
        int(incumbent["early_stopping_rounds"]),
        None,
    )
    print(f"[Wave 0] Training official RandomForest for validation {validation_season}...")
    rf_result = train_rf_holdout(train, train_idx, valid_idx, official_features)
    lgb_raw = np.asarray(lgb_result["prediction"], dtype=np.float64)
    rf_raw = np.asarray(rf_result["prediction"], dtype=np.float64)
    lgb_weight = float(incumbent["lgb_weight"])
    rf_weight = float(incumbent["rf_weight"])
    blend_raw = lgb_weight * lgb_raw + rf_weight * rf_raw

    rates = _season_rates(train, train_idx)
    trend_window = min(int(incumbent["trend_window"]), len(rates))
    if trend_window >= 2:
        offset, forecast_rate, _ = fit_season_logit_offset(
            train.iloc[train_idx]["season"].to_numpy(),
            train.iloc[train_idx][TARGET_COL].to_numpy(),
            validation_season,
            trend_window,
        )
    else:
        offset, forecast_rate = 0.0, float(rates.iloc[-1])
    lgb_trend = apply_logit_offset(lgb_raw, offset)
    rf_trend = apply_logit_offset(rf_raw, offset)
    incumbent_prediction = lgb_weight * lgb_trend + rf_weight * rf_trend
    np.savez_compressed(
        cache_path,
        valid_idx=valid_idx.astype(np.int64),
        target=target,
        lgb_raw=lgb_raw,
        rf_raw=rf_raw,
        blend_raw=blend_raw,
        lgb_trend=lgb_trend,
        rf_trend=rf_trend,
        incumbent=incumbent_prediction,
        offset=np.array([offset], dtype=np.float64),
        forecast_rate=np.array([forecast_rate], dtype=np.float64),
        trend_window=np.array([trend_window], dtype=np.int16),
        lgb_best_iteration=np.array([lgb_result["best_iteration"]], dtype=np.int32),
        lgb_fit_seconds=np.array([lgb_result["fit_seconds"]], dtype=np.float64),
        rf_fit_seconds=np.array([rf_result["fit_seconds"]], dtype=np.float64),
        lgb_inference_seconds=np.array([lgb_result["inference_seconds"]], dtype=np.float64),
        rf_inference_seconds=np.array([rf_result["inference_seconds"]], dtype=np.float64),
        peak_memory_mb=np.array(
            [max(lgb_result["peak_memory_mb"], rf_result["peak_memory_mb"])],
            dtype=np.float64,
        ),
    )
    del lgb_result, rf_result
    gc.collect()
    return _load_fold_cache(cache_path, valid_idx, target)


def _segment_rows(
    frame: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    validation_season: int,
) -> list[dict[str, Any]]:
    work = frame.copy()
    work["__target"] = target
    work["__prediction"] = prediction
    work["__sq_error"] = np.square(prediction - target)
    work["pitcher_sample_bucket"] = pd.cut(
        pd.to_numeric(work["asof_pitcher_n"]),
        bins=[-1, 0, 29, 99, 299, np.inf],
        labels=["0", "1-29", "30-99", "100-299", "300+"],
    ).astype("string")
    work["batter_sample_bucket"] = pd.cut(
        pd.to_numeric(work["asof_batter_n"]),
        bins=[-1, 0, 29, 99, 299, np.inf],
        labels=["0", "1-29", "30-99", "100-299", "300+"],
    ).astype("string")
    work["count_state"] = (
        work["balls_before"].astype("string") + "-" + work["strikes_before"].astype("string")
    )
    group_columns = [
        "pitcher_sample_bucket",
        "batter_sample_bucket",
        "game_type",
        "count_state",
        "inning",
        "base_state",
        "pitcher_hand",
        "batter_hand",
    ]
    rows: list[dict[str, Any]] = []
    for column in group_columns:
        for value, group in work.groupby(column, observed=True, dropna=False):
            rows.append(
                {
                    "experiment_id": "W0_incumbent",
                    "outer_validation_season": validation_season,
                    "segment_type": column,
                    "segment_value": str(value),
                    "n_rows": len(group),
                    "brier": float(group["__sq_error"].mean()),
                    "prediction_mean": float(group["__prediction"].mean()),
                    "target_rate": float(group["__target"].mean()),
                }
            )
    return rows


def run_wave0(project_dir: Path, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    started = time.perf_counter()
    print("[Wave 0] Loading train data...")
    train = read_main(project_dir / "data" / "train.csv")
    test_columns = read_main(project_dir / "data" / "test.csv", nrows=0).columns.tolist()
    official_features = [column for column in test_columns if column != ID_COL]
    years = [int(config["diagnostic_validation_season"])] + [
        int(year) for year in config["outer_validation_seasons"]
    ]
    folds = {
        fold.validation_season: fold
        for fold in walk_forward_splits(train, validation_seasons=tuple(years))
    }
    caches: dict[int, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    reliability_rows: list[pd.DataFrame] = []
    segment_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    experiments_path = project_dir / "reports" / "experiments.csv"

    for year in years:
        fold = folds[year]
        cached = _generate_wave0_fold(
            project_dir,
            train,
            fold.train_idx,
            fold.valid_idx,
            year,
            config,
            official_features,
        )
        caches[year] = cached
        target = cached["target"]
        incumbent_prediction = cached["incumbent"]
        predictions = {
            "lgb_raw": cached["lgb_raw"],
            "rf_raw": cached["rf_raw"],
            "blend_raw": cached["blend_raw"],
            "lgb_trend": cached["lgb_trend"],
            "rf_trend": cached["rf_trend"],
            "incumbent_fixed_blend_trend": incumbent_prediction,
        }
        diagnostic = year == int(config["diagnostic_validation_season"])
        for model, prediction in predictions.items():
            result_rows.append(
                _metric_row(
                    f"W0_{model}_{year}",
                    year,
                    model,
                    prediction,
                    target,
                    incumbent_prediction,
                    diagnostic=diagnostic,
                    prediction_source="strict outer train; fixed incumbent recipe",
                )
            )
            for bins in (10, 20):
                decomposition = brier_decomposition(target, prediction, n_bins=bins)
                decomposition_rows.append(
                    {
                        "experiment_id": f"W0_{model}_{year}",
                        "outer_validation_season": year,
                        "model": model,
                        **decomposition,
                    }
                )
            table = reliability_table(target, prediction, n_bins=20)
            table.insert(0, "model", model)
            table.insert(0, "outer_validation_season", year)
            table.insert(0, "experiment_id", f"W0_{model}_{year}")
            reliability_rows.append(table)
        correlation_rows.append(
            {
                "outer_validation_season": year,
                "lgb_rf_correlation": float(np.corrcoef(cached["lgb_raw"], cached["rf_raw"])[0, 1]),
                "trend_offset": float(cached["offset"][0]),
                "trend_window": int(cached["trend_window"][0]),
                "trend_forecast_rate": float(cached["forecast_rate"][0]),
                "blend_raw_brier": brier_score(target, cached["blend_raw"]),
                "incumbent_brier": brier_score(target, incumbent_prediction),
                "trend_delta_brier": brier_score(target, incumbent_prediction)
                - brier_score(target, cached["blend_raw"]),
            }
        )
        valid_frame = train.iloc[fold.valid_idx].reset_index(drop=True)
        segment_rows.extend(_segment_rows(valid_frame, target, incumbent_prediction, year))
        pitcher_season = (
            valid_frame["pitcher_id"].astype("string")
            + "-"
            + valid_frame["season"].astype("string")
        )
        for cluster_name, clusters in (
            ("pitcher_season", pitcher_season),
            ("pitcher", valid_frame["pitcher_id"].astype("string")),
        ):
            bootstrap_rows.append(
                {
                    "experiment_id": f"W0_trend_vs_raw_{year}",
                    "outer_validation_season": year,
                    "candidate": "incumbent_fixed_blend_trend",
                    "reference": "blend_raw",
                    "cluster_type": cluster_name,
                    **cluster_bootstrap_delta(
                        target,
                        incumbent_prediction,
                        cached["blend_raw"],
                        clusters,
                        n_resamples=int(config["bootstrap_resamples"]),
                        seed=int(config["seed"]) + year,
                    ),
                }
            )

        _append_experiment_once(
            experiments_path,
            {
                "experiment_id": f"W0_incumbent_walk_forward_{year}",
                "validation_split": f"train_before_{year}_validate_{year}",
                "feature_groups": "incumbent engineered LGB + official RF + train-only season trend",
                "model_params": json.dumps(config["incumbent"], sort_keys=True),
                "seed": config["seed"],
                "raw_brier_score": f"{brier_score(target, incumbent_prediction):.9f}",
                "local_brier_skill_score": f"{brier_skill_score(target, incumbent_prediction):.3f}",
                "calibration": f"train-only logit trend window={int(cached['trend_window'][0])}",
                "training_seconds": f"{float(cached['lgb_fit_seconds'][0] + cached['rf_fit_seconds'][0]):.3f}",
                "inference_seconds": f"{float(cached['lgb_inference_seconds'][0] + cached['rf_inference_seconds'][0]):.3f}",
                "peak_memory_mb": f"{float(cached['peak_memory_mb'][0]):.1f}",
                "leakage_risk_notes": "strict prior-season outer train; fixed 0.35/0.65 weights; season used only by incumbent-compatible recipe and train-only trend",
            },
        )
        print(
            f"[Wave 0] {year}: incumbent={brier_score(target, incumbent_prediction):.9f}, "
            f"raw_blend={brier_score(target, cached['blend_raw']):.9f}"
        )

    _upsert_csv(
        project_dir / "reports" / "walk_forward_results.csv",
        pd.DataFrame(result_rows),
        ["experiment_id"],
    )
    _upsert_csv(
        project_dir / "reports" / "brier_decomposition.csv",
        pd.DataFrame(decomposition_rows),
        ["experiment_id", "n_bins"],
    )
    _upsert_csv(
        project_dir / "reports" / "reliability_tables.csv",
        pd.concat(reliability_rows, ignore_index=True),
        ["experiment_id", "bin"],
    )
    _upsert_csv(
        project_dir / "reports" / "segment_results.csv",
        pd.DataFrame(segment_rows),
        ["experiment_id", "outer_validation_season", "segment_type", "segment_value"],
    )
    _write_csv(project_dir / "reports" / "wave0_correlations.csv", pd.DataFrame(correlation_rows))
    _upsert_csv(
        project_dir / "reports" / "bootstrap_results.csv",
        pd.DataFrame(bootstrap_rows),
        ["experiment_id", "cluster_type"],
    )
    primary = pd.DataFrame(result_rows)
    primary = primary[
        (primary["validation_role"] == "primary")
        & (primary["model"] == "incumbent_fixed_blend_trend")
    ].copy()
    weights = np.array(
        [float(config["recency_weights"][str(year)]) for year in primary["outer_validation_season"]]
    )
    summary = {
        "runtime_seconds": time.perf_counter() - started,
        "mean_brier": float(primary["brier"].mean()),
        "recency_weighted_brier": float(np.average(primary["brier"], weights=weights)),
        "fold_std": float(primary["brier"].std(ddof=0)),
        "worst_fold_brier": float(primary["brier"].max()),
        "folds": {
            str(int(row.outer_validation_season)): float(row.brier)
            for row in primary.itertuples()
        },
    }
    summary_path = project_dir / "artifacts" / "followup" / "wave0_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Wave 0] Completed in {summary['runtime_seconds']:.1f}s: {summary}")
    del train
    gc.collect()
    return caches


def _load_all_caches(project_dir: Path, config: dict[str, Any], train: pd.DataFrame):
    years = [int(config["diagnostic_validation_season"])] + [
        int(year) for year in config["outer_validation_seasons"]
    ]
    folds = {
        fold.validation_season: fold
        for fold in walk_forward_splits(train, validation_seasons=tuple(years))
    }
    caches = {}
    for year in years:
        fold = folds[year]
        path = _fold_cache_path(project_dir, year)
        if not path.exists():
            raise FileNotFoundError(f"Wave 0 cache missing for {year}: {path}")
        caches[year] = _load_fold_cache(
            path,
            fold.valid_idx,
            train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=np.int8),
        )
    return folds, caches


def _fit_calibrator(method: str, prediction: np.ndarray, target: np.ndarray, config: dict[str, Any]):
    if method == "intercept":
        return InterceptCalibrator().fit(prediction, target)
    if method == "platt":
        return PlattCalibrator().fit(prediction, target)
    if method.startswith("beta_"):
        regularization = float(method.split("_", 1)[1])
        return BetaCalibrator(regularization=regularization).fit(prediction, target)
    if method == "isotonic":
        return IsotonicCalibrator().fit(prediction, target)
    raise ValueError(method)


def run_wave1(project_dir: Path, config: dict[str, Any]) -> None:
    started = time.perf_counter()
    print("[Wave 1] Loading train data and Wave 0 caches...")
    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, config, train)
    primary_years = [int(year) for year in config["outer_validation_seasons"]]
    result_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    method_history: dict[str, list[tuple[int, float]]] = {
        method: [] for method in config["base_rate_methods"]
    }

    for year in [int(config["diagnostic_validation_season"])] + primary_years:
        fold = folds[year]
        target = caches[year]["target"]
        incumbent_prediction = caches[year]["incumbent"]
        raw_blend = caches[year]["blend_raw"]
        rates = _season_rates(train, fold.train_idx)
        available_predictions: dict[str, np.ndarray] = {}
        for method in config["base_rate_methods"]:
            try:
                forecast_rate = forecast_base_rate(rates, year, method)
            except ValueError:
                continue
            latest_rate = float(rates.iloc[-1])
            offset = float(_logit(np.array([forecast_rate]))[0] - _logit(np.array([latest_rate]))[0])
            prediction = apply_logit_offset(raw_blend, offset)
            available_predictions[method] = prediction
            score = brier_score(target, prediction)
            forecast_rows.append(
                {
                    "outer_validation_season": year,
                    "method": method,
                    "history_seasons": ";".join(str(int(value)) for value in rates.index),
                    "latest_rate": latest_rate,
                    "forecast_rate": forecast_rate,
                    "offset": offset,
                    "brier": score,
                    "delta_brier_vs_incumbent": score
                    - brier_score(target, incumbent_prediction),
                }
            )
            result_rows.append(
                _metric_row(
                    f"W1_drift_{method}_{year}",
                    year,
                    f"drift_{method}",
                    prediction,
                    target,
                    incumbent_prediction,
                    diagnostic=year not in primary_years,
                    prediction_source="raw incumbent blend plus train-only season-rate forecast",
                )
            )
        if available_predictions:
            simple_average = np.mean(np.column_stack(list(available_predictions.values())), axis=1)
            result_rows.append(
                _metric_row(
                    f"W1_drift_simple_average_{year}",
                    year,
                    "drift_simple_average",
                    simple_average,
                    target,
                    incumbent_prediction,
                    diagnostic=year not in primary_years,
                    prediction_source="equal average of all available predeclared train-only forecasts",
                )
            )

        historical_years = [
            prior_year
            for prior_year in sorted(caches)
            if prior_year < year
        ]
        if historical_years:
            inner_prediction = np.concatenate(
                [caches[prior_year]["incumbent"] for prior_year in historical_years]
            )
            inner_target = np.concatenate(
                [caches[prior_year]["target"] for prior_year in historical_years]
            )
            calibration_methods = ["intercept", "platt"] + [
                f"beta_{value}" for value in config["beta_regularization"]
            ]
            if len(historical_years) >= 2:
                calibration_methods.append("isotonic")
            for method in calibration_methods:
                calibrator = _fit_calibrator(method, inner_prediction, inner_target, config)
                prediction = calibrator.predict(incumbent_prediction)
                score = brier_score(target, prediction)
                calibration_rows.append(
                    {
                        "experiment_id": f"W1_calibration_{method}_{year}",
                        "outer_validation_season": year,
                        "method": method,
                        "fit_seasons": ";".join(str(value) for value in historical_years),
                        "fit_rows": len(inner_target),
                        "parameters": json.dumps(calibrator.to_dict(), sort_keys=True),
                        "raw_brier": brier_score(target, incumbent_prediction),
                        "calibrated_brier": score,
                        "delta_brier": score - brier_score(target, incumbent_prediction),
                    }
                )
                result_rows.append(
                    _metric_row(
                        f"W1_calibration_{method}_{year}",
                        year,
                        f"calibration_{method}",
                        prediction,
                        target,
                        incumbent_prediction,
                        diagnostic=year not in primary_years,
                        prediction_source="calibrator fit on earlier forward OOF only",
                        fit_seasons=";".join(str(value) for value in historical_years),
                    )
                )

    _upsert_csv(
        project_dir / "reports" / "walk_forward_results.csv",
        pd.DataFrame(result_rows),
        ["experiment_id"],
    )
    _upsert_csv(
        project_dir / "reports" / "calibration_results.csv",
        pd.DataFrame(calibration_rows),
        ["experiment_id"],
    )
    _write_csv(project_dir / "reports" / "base_rate_forecasts.csv", pd.DataFrame(forecast_rows))
    print(f"[Wave 1] Completed in {time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--waves",
        nargs="+",
        choices=["0", "1"],
        default=["0", "1"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/followup.json"),
    )
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "0" in args.waves:
        run_wave0(project_dir, config)
    if "1" in args.waves:
        run_wave1(project_dir, config)


if __name__ == "__main__":
    main()
