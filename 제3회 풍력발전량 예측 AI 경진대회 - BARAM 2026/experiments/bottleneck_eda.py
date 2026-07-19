from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.cross_group_trajectory_smoothing import (
    _current_cross_group_prediction,
    smooth_group_3,
)
from experiments.cross_group_transfer import (
    fit_predict_models,
    selected_prediction,
    transfer_features,
)
from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
H2_START = pd.Timestamp("2024-07-01 01:00:00")
RATIO_EDGES = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01])
WIND_EDGES = np.asarray([0.0, 6.0, 8.0, 10.0, 12.0, 15.0, np.inf])
KEY_SHIFT_FEATURES = (
    "ldaps__kpx_group_3__hub_ws117__idw",
    "ldaps__hub_ws117__grid_8",
    "ldaps__hub_ws117__grid_13",
    "gfs__kpx_group_3__hub_ws117__idw",
    "gfs__kpx_group_3__surface_0_gust__idw",
    "ldaps__kpx_group_3__hub_dir_sin__idw",
    "ldaps__kpx_group_3__hub_dir_cos__idw",
)


def population_stability_index(
    reference: np.ndarray,
    comparison: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate PSI with bins learned only from the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]
    if len(reference) == 0 or len(comparison) == 0:
        raise ValueError("PSI requires finite values in both samples")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")

    internal = np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    internal = np.unique(internal)
    if len(internal) == 0:
        return 0.0 if np.all(comparison == reference[0]) else float("inf")
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    reference_fraction = np.histogram(reference, bins=edges)[0] / len(reference)
    comparison_fraction = np.histogram(comparison, bins=edges)[0] / len(comparison)
    epsilon = 1e-6
    reference_fraction = np.clip(reference_fraction, epsilon, None)
    comparison_fraction = np.clip(comparison_fraction, epsilon, None)
    return float(
        np.sum(
            (comparison_fraction - reference_fraction)
            * np.log(comparison_fraction / reference_fraction)
        )
    )


def stable_binary_auc(
    values_1: np.ndarray,
    target_1: np.ndarray,
    values_2: np.ndarray,
    target_2: np.ndarray,
) -> dict[str, float | int]:
    """Return same-direction AUCs, orienting the sign on the first period."""
    values_1 = np.asarray(values_1, dtype=float)
    values_2 = np.asarray(values_2, dtype=float)
    target_1 = np.asarray(target_1, dtype=int)
    target_2 = np.asarray(target_2, dtype=int)
    mask_1 = np.isfinite(values_1)
    mask_2 = np.isfinite(values_2)
    if len(np.unique(target_1[mask_1])) < 2 or len(np.unique(target_2[mask_2])) < 2:
        raise ValueError("AUC requires both classes in both periods")
    auc_1 = float(roc_auc_score(target_1[mask_1], values_1[mask_1]))
    auc_2 = float(roc_auc_score(target_2[mask_2], values_2[mask_2]))
    direction = 1 if auc_1 >= 0.5 else -1
    oriented_1 = auc_1 if direction == 1 else 1.0 - auc_1
    oriented_2 = auc_2 if direction == 1 else 1.0 - auc_2
    return {
        "direction": direction,
        "auc_period_1": oriented_1,
        "auc_period_2": oriented_2,
        "stable_auc": min(oriented_1, oriented_2),
    }


def ratio_bin_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
    edges: np.ndarray = RATIO_EDGES,
    by_prediction: bool = False,
) -> list[dict[str, float | int | str]]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    source = prediction if by_prediction else truth
    ratio = source / capacity
    rows: list[dict[str, float | int | str]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (
            np.isfinite(truth)
            & np.isfinite(prediction)
            & (truth >= 0.10 * capacity)
            & (ratio >= low)
            & (ratio < high)
        )
        if not mask.any():
            continue
        metric = evaluate_group(truth[mask], prediction[mask], capacity)
        error = prediction[mask] - truth[mask]
        rows.append(
            {
                "bin": f"{low:.1f}-{high:.2g}",
                "n": int(mask.sum()),
                "score": metric.score,
                "one_minus_nmae": metric.one_minus_nmae,
                "ficr": metric.ficr,
                "bias_kwh": float(error.mean()),
                "mae_kwh": float(np.abs(error).mean()),
                "miss_8pct_fraction": float(np.mean(np.abs(error) > 0.08 * capacity)),
            }
        )
    return rows


def _metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return evaluate_group(truth, prediction, CAPACITY_KWH[TARGET]).to_dict()


def _load_labels(data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    return labels.set_index("kst_dtm").sort_index()


def _label_distribution(labels: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows = []
    for target, capacity in CAPACITY_KWH.items():
        for year in (2022, 2023, 2024):
            values = labels.loc[labels.index.year == year, target]
            observed = values.dropna()
            row: dict[str, float | int | str] = {
                "target": target,
                "year": year,
                "rows": int(len(values)),
                "observed": int(len(observed)),
                "missing": int(values.isna().sum()),
            }
            if len(observed):
                normalized = observed / capacity
                row.update(
                    {
                        "eligible_fraction": float((normalized >= 0.10).mean()),
                        "mean_ratio": float(normalized.mean()),
                        "p10_ratio": float(normalized.quantile(0.10)),
                        "median_ratio": float(normalized.median()),
                        "p90_ratio": float(normalized.quantile(0.90)),
                        "high_power_fraction": float((normalized >= 0.80).mean()),
                    }
                )
            rows.append(row)
    return rows


def _cross_group_member(
    labels: pd.DataFrame,
    weighted: np.lib.npyio.NpzFile,
    keep: np.ndarray,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_1 = selected_prediction(weighted, "kpx_group_1")[keep]
    group_2 = selected_prediction(weighted, "kpx_group_2")[keep]
    group_1_ratio = group_1 / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = group_2 / CAPACITY_KWH["kpx_group_2"]
    train = labels[labels.index.year == 2023].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    member = fit_predict_models(
        transfer_features(
            train["kpx_group_1"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_1"],
            train["kpx_group_2"].to_numpy(dtype=float) / CAPACITY_KWH["kpx_group_2"],
            train.index,
        ),
        train[TARGET].to_numpy(dtype=float),
        transfer_features(group_1_ratio, group_2_ratio, index),
        seed=51_000,
        mode="base",
    )
    return group_1, group_2, np.clip(member, 0.0, CAPACITY_KWH[TARGET])


def _proxy_predictions(
    labels: pd.DataFrame,
    cache_paths: dict[str, Path],
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    weighted = np.load(cache_paths["weighted"], allow_pickle=True)
    full_index = pd.DatetimeIndex(pd.to_datetime(weighted[f"{TARGET}__valid_index_ns"]))
    keep = full_index.year == 2024
    index = full_index[keep]
    group_1, group_2, member = _cross_group_member(labels, weighted, keep, index)
    group_1_ratio = group_1 / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = group_2 / CAPACITY_KWH["kpx_group_2"]
    predictions = {}
    for name, path in cache_paths.items():
        cache = np.load(path, allow_pickle=True)
        if not np.array_equal(
            cache[f"{TARGET}__valid_index_ns"], weighted[f"{TARGET}__valid_index_ns"]
        ):
            raise ValueError(f"Validation index mismatch for {name}")
        raw = selected_prediction(cache, TARGET)[keep]
        current = _current_cross_group_prediction(
            raw, member, group_1_ratio, group_2_ratio
        )
        current, _ = smooth_group_3(
            group_1, group_2, current, index, alpha=0.05, max_delta_ratio=0.02
        )
        predictions[name] = current
    truth = labels.reindex(index)[TARGET].to_numpy(dtype=float)
    return index, truth, predictions


def _wind_regimes(
    features: pd.DataFrame,
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, list[dict[str, float | int | str]]]:
    frame = features.reindex(index)
    wind = frame["ldaps__kpx_group_3__hub_ws117__idw"].to_numpy(dtype=float)
    direction = (
        np.degrees(
            np.arctan2(
                frame["ldaps__kpx_group_3__hub_dir_sin__idw"],
                frame["ldaps__kpx_group_3__hub_dir_cos__idw"],
            )
        )
        + 360.0
    ) % 360.0
    eligible = truth >= 0.10 * CAPACITY_KWH[TARGET]

    wind_rows = []
    for low, high in zip(WIND_EDGES[:-1], WIND_EDGES[1:]):
        mask = eligible & (wind >= low) & (wind < high)
        metric = _metric_record(truth[mask], prediction[mask])
        wind_rows.append(
            {
                "wind_bin": f"{low:g}-{high:g}",
                "n": int(mask.sum()),
                "truth_mean_ratio": float(np.mean(truth[mask]) / CAPACITY_KWH[TARGET]),
                "high_power_fraction": float(
                    np.mean(truth[mask] >= 0.80 * CAPACITY_KWH[TARGET])
                ),
                "bias_kwh": float(np.mean(prediction[mask] - truth[mask])),
                **metric,
            }
        )

    direction_rows = []
    for low in range(0, 360, 45):
        mask = eligible & (wind >= 10.0) & (direction >= low) & (direction < low + 45)
        if not mask.any():
            continue
        direction_rows.append(
            {
                "direction_bin": f"{low}-{low + 45}",
                "n": int(mask.sum()),
                "truth_mean_ratio": float(np.mean(truth[mask]) / CAPACITY_KWH[TARGET]),
                "high_power_fraction": float(
                    np.mean(truth[mask] >= 0.80 * CAPACITY_KWH[TARGET])
                ),
                "bias_kwh": float(np.mean(prediction[mask] - truth[mask])),
                **_metric_record(truth[mask], prediction[mask]),
            }
        )

    month_rows = []
    for month in range(1, 13):
        mask = eligible & (index.month == month)
        error = prediction[mask] - truth[mask]
        month_rows.append(
            {
                "month": month,
                "n": int(mask.sum()),
                "bias_kwh": float(error.mean()),
                "mae_kwh": float(np.abs(error).mean()),
                **_metric_record(truth[mask], prediction[mask]),
            }
        )

    strong_wind_month_rows = []
    observed = np.isfinite(truth)
    for month in range(1, 13):
        mask = observed & (index.month == month) & (wind >= 10.0)
        normalized = truth[mask] / CAPACITY_KWH[TARGET]
        strong_wind_month_rows.append(
            {
                "month": month,
                "n": int(mask.sum()),
                "truth_mean_ratio": float(normalized.mean()),
                "eligible_fraction": float(np.mean(normalized >= 0.10)),
                "high_power_fraction": float(np.mean(normalized >= 0.80)),
                "below_half_fraction": float(np.mean(normalized < 0.50)),
            }
        )
    return {
        "wind_speed": wind_rows,
        "strong_wind_direction": direction_rows,
        "month": month_rows,
        "strong_wind_month": strong_wind_month_rows,
    }


def _feature_shift(features: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for column in KEY_SHIFT_FEATURES:
        if column not in features:
            continue
        by_year = {
            year: features.loc[features.index.year == year, column].dropna().to_numpy(dtype=float)
            for year in (2023, 2024, 2025)
        }
        stats = {}
        for year, values in by_year.items():
            stats[str(year)] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
            }
        rows.append(
            {
                "feature": column,
                "stats": stats,
                "psi_2023_to_2024": population_stability_index(by_year[2023], by_year[2024]),
                "psi_2024_to_2025": population_stability_index(by_year[2024], by_year[2025]),
            }
        )
    return rows


def _high_power_features(
    train_features: pd.DataFrame,
    labels: pd.DataFrame,
    top_n: int,
) -> list[dict[str, float | int | str]]:
    joined = labels[TARGET].reindex(train_features.index)
    masks = {
        year: (train_features.index.year == year) & joined.notna()
        for year in (2023, 2024)
    }
    targets = {
        year: (
            joined.loc[masks[year]].to_numpy(dtype=float)
            >= 0.80 * CAPACITY_KWH[TARGET]
        ).astype(int)
        for year in (2023, 2024)
    }
    rows = []
    for column in train_features.columns:
        try:
            result = stable_binary_auc(
                train_features.loc[masks[2023], column].to_numpy(dtype=float),
                targets[2023],
                train_features.loc[masks[2024], column].to_numpy(dtype=float),
                targets[2024],
            )
        except ValueError:
            continue
        rows.append({"feature": column, **result})
    rows.sort(key=lambda row: float(row["stable_auc"]), reverse=True)
    return rows[:top_n]


def analyze(
    data_dir: Path,
    cache_dir: Path,
    cache_paths: dict[str, Path],
    top_n: int = 20,
) -> dict[str, object]:
    labels = _load_labels(data_dir)
    train_features = load_or_build_features(data_dir, "train", cache_dir)
    test_features = load_or_build_features(data_dir, "test", cache_dir)
    features = pd.concat([train_features, test_features]).sort_index()
    index, truth, proxies = _proxy_predictions(labels, cache_paths)
    capacity = CAPACITY_KWH[TARGET]
    proxy_metrics = {name: _metric_record(truth, pred) for name, pred in proxies.items()}
    spread = np.std(np.column_stack(list(proxies.values())), axis=1) / capacity
    weighted = proxies["weighted"]
    eligible = truth >= 0.10 * capacity
    report: dict[str, object] = {
        "generated_for": "BARAM 2026 group-3 bottleneck after submission 1491284",
        "validation": {
            "start": str(index.min()),
            "end": str(index.max()),
            "rows": int(len(index)),
            "eligible_rows": int(eligible.sum()),
            "h2_start": str(H2_START),
        },
        "label_distribution": _label_distribution(labels),
        "proxy_metrics": proxy_metrics,
        "proxy_spread": {
            "median_ratio": float(np.median(spread)),
            "p90_ratio": float(np.quantile(spread, 0.90)),
            "p95_ratio": float(np.quantile(spread, 0.95)),
            "correlation_with_absolute_error": float(
                np.corrcoef(spread[eligible], np.abs(weighted[eligible] - truth[eligible]))[0, 1]
            ),
        },
        "weighted_proxy_error_by_truth_ratio": ratio_bin_diagnostics(
            truth, weighted, capacity
        ),
        "weighted_proxy_error_by_prediction_ratio": ratio_bin_diagnostics(
            truth, weighted, capacity, by_prediction=True
        ),
        "weighted_proxy_regimes": _wind_regimes(
            train_features, index, truth, weighted
        ),
        "feature_shift": _feature_shift(features),
        "stable_high_power_features": _high_power_features(
            train_features, labels, top_n
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts_feature_cache")
    parser.add_argument("--artifact-dir", default="artifacts_bottleneck_eda")
    parser.add_argument(
        "--weighted-cache", default="artifacts_weighted_metric/prediction_cache.npz"
    )
    parser.add_argument("--global-cache", default="artifacts_global/prediction_cache.npz")
    parser.add_argument("--pool-cache", default="artifacts_final_pool/prediction_cache.npz")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    report = analyze(
        Path(args.data_dir),
        Path(args.cache_dir),
        {
            "weighted": Path(args.weighted_cache),
            "global": Path(args.global_cache),
            "pool": Path(args.pool_cache),
        },
        top_n=args.top_n,
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "bottleneck_eda_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "proxy_metrics": report["proxy_metrics"],
                "proxy_spread": report["proxy_spread"],
                "top_high_power_features": report["stable_high_power_features"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
