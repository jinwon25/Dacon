from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cross_group_transfer import (
    fit_predict_models,
    selected_prediction,
    transfer_features,
)
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
TARGET = "kpx_group_3"


def shift_within_issue_cycle(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    offset_hours: int,
) -> np.ndarray:
    """Shift a forecast trajectory without borrowing from another issue cycle.

    BARAM forecast cycles contain target hours 01:00 through 00:00 the next day.
    Missing neighbors and cycle boundaries fall back to the current prediction.
    """
    values = np.asarray(values, dtype=float)
    timestamps = pd.DatetimeIndex(timestamps)
    if values.ndim != 1 or len(values) != len(timestamps):
        raise ValueError("values and timestamps must be aligned one-dimensional arrays")
    if timestamps.has_duplicates:
        raise ValueError("timestamps must be unique")
    if not np.isfinite(values).all():
        raise ValueError("trajectory values must be finite")

    query = timestamps + pd.to_timedelta(offset_hours, unit="h")
    cycle = (timestamps - pd.Timedelta(hours=1)).normalize()
    query_cycle = (query - pd.Timedelta(hours=1)).normalize()
    shifted = pd.Series(values, index=timestamps).reindex(query).to_numpy(dtype=float)
    usable = np.asarray(cycle == query_cycle) & np.isfinite(shifted)
    return np.where(usable, shifted, values)


def triangular_smooth(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    previous = shift_within_issue_cycle(values, timestamps, -1)
    following = shift_within_issue_cycle(values, timestamps, 1)
    return (previous + 2.0 * np.asarray(values, dtype=float) + following) / 4.0


def driver_consensus_mask(
    group_1: np.ndarray,
    group_2: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Keep rows where both driver farms imply the same smoothing direction."""
    group_1_ratio = np.asarray(group_1, dtype=float) / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = np.asarray(group_2, dtype=float) / CAPACITY_KWH["kpx_group_2"]
    delta_1 = triangular_smooth(group_1_ratio, timestamps) - group_1_ratio
    delta_2 = triangular_smooth(group_2_ratio, timestamps) - group_2_ratio
    return delta_1 * delta_2 >= 0.0


def smooth_group_3(
    group_1: np.ndarray,
    group_2: np.ndarray,
    group_3: np.ndarray,
    timestamps: pd.DatetimeIndex,
    alpha: float = 0.05,
    max_delta_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if max_delta_ratio < 0.0:
        raise ValueError("max_delta_ratio must be non-negative")

    group_3 = np.asarray(group_3, dtype=float)
    smoothed = triangular_smooth(group_3, timestamps)
    mask = driver_consensus_mask(group_1, group_2, timestamps)
    mask &= np.abs(smoothed - group_3) / CAPACITY_KWH[TARGET] <= max_delta_ratio
    output = group_3.copy()
    output[mask] = (1.0 - alpha) * group_3[mask] + alpha * smoothed[mask]
    return np.clip(output, 0.0, CAPACITY_KWH[TARGET]), mask


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _current_cross_group_prediction(
    base: np.ndarray,
    member: np.ndarray,
    group_1_ratio: np.ndarray,
    group_2_ratio: np.ndarray,
) -> np.ndarray:
    capacity = CAPACITY_KWH[TARGET]
    gate = (
        (np.abs(group_1_ratio - group_2_ratio) <= 0.08)
        & (np.abs(member - base) / capacity <= 0.06)
        & (base >= 0.10 * capacity)
    )
    output = np.asarray(base, dtype=float).copy()
    output[gate] = 0.75 * output[gate] + 0.25 * member[gate]
    return output


def validate_policy(
    labels: pd.DataFrame,
    weighted_cache_path: Path,
    proxy_cache_paths: dict[str, Path],
    alpha: float,
    max_delta_ratio: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    weighted = np.load(weighted_cache_path, allow_pickle=True)
    all_index = pd.to_datetime(weighted[f"{TARGET}__valid_index_ns"])
    keep = all_index.year == 2024
    timestamps = all_index[keep]
    group_1 = selected_prediction(weighted, "kpx_group_1")[keep]
    group_2 = selected_prediction(weighted, "kpx_group_2")[keep]
    group_1_ratio = group_1 / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = group_2 / CAPACITY_KWH["kpx_group_2"]
    truth = labels.reindex(timestamps)[TARGET].to_numpy(dtype=float)

    transfer_train = labels[labels.index.year == 2023].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    train_features = transfer_features(
        transfer_train["kpx_group_1"].to_numpy(dtype=float)
        / CAPACITY_KWH["kpx_group_1"],
        transfer_train["kpx_group_2"].to_numpy(dtype=float)
        / CAPACITY_KWH["kpx_group_2"],
        transfer_train.index,
    )
    member = np.clip(
        fit_predict_models(
            train_features,
            transfer_train[TARGET].to_numpy(dtype=float),
            transfer_features(group_1_ratio, group_2_ratio, timestamps),
            seed=51_000,
            mode="base",
        ),
        0.0,
        CAPACITY_KWH[TARGET],
    )

    h1 = timestamps < pd.Timestamp("2024-07-01 01:00:00")
    periods = {"h1": h1, "h2": ~h1, "full": np.ones(len(timestamps), dtype=bool)}
    rng = np.random.default_rng(seed)
    report: dict[str, object] = {
        "policy": {
            "alpha": alpha,
            "max_delta_ratio": max_delta_ratio,
            "driver_consensus": "group-1 and group-2 triangular deltas have the same sign",
            "issue_cycle_boundary": "01:00 through next-day 00:00",
        },
        "validation_rows": int(len(timestamps)),
        "h2_start": "2024-07-01 01:00:00",
        "bootstrap_splits": n_bootstrap,
        "proxies": {},
    }

    for name, cache_path in proxy_cache_paths.items():
        proxy = np.load(cache_path, allow_pickle=True)
        if not np.array_equal(
            proxy[f"{TARGET}__valid_index_ns"], weighted[f"{TARGET}__valid_index_ns"]
        ):
            raise ValueError(f"Validation index mismatch for {name}")
        pre_cross = selected_prediction(proxy, TARGET)[keep]
        current = _current_cross_group_prediction(
            pre_cross, member, group_1_ratio, group_2_ratio
        )
        candidate, mask = smooth_group_3(
            group_1, group_2, current, timestamps, alpha, max_delta_ratio
        )

        period_report = {}
        for period_name, period_mask in periods.items():
            before = evaluate_group(
                truth[period_mask], current[period_mask], CAPACITY_KWH[TARGET]
            )
            after = evaluate_group(
                truth[period_mask], candidate[period_mask], CAPACITY_KWH[TARGET]
            )
            period_report[period_name] = _metric_delta(before, after)

        monthly_deltas = []
        for month in range(1, 13):
            month_mask = timestamps.month == month
            before = evaluate_group(
                truth[month_mask], current[month_mask], CAPACITY_KWH[TARGET]
            )
            after = evaluate_group(
                truth[month_mask], candidate[month_mask], CAPACITY_KWH[TARGET]
            )
            monthly_deltas.append(
                {"month": month, **_metric_delta(before, after)}
            )

        bootstrap_deltas = []
        for _ in range(n_bootstrap):
            private = ~(rng.random(len(timestamps)) < 0.40)
            before = evaluate_group(
                truth[private], current[private], CAPACITY_KWH[TARGET]
            )
            after = evaluate_group(
                truth[private], candidate[private], CAPACITY_KWH[TARGET]
            )
            bootstrap_deltas.append(list(_metric_delta(before, after).values()))
        bootstrap = np.asarray(bootstrap_deltas, dtype=float)
        metric_names = ["score", "one_minus_nmae", "ficr"]

        report["proxies"][name] = {
            "changed_rows": int(mask.sum()),
            "mean_absolute_movement": float(np.mean(np.abs(candidate - current))),
            "period_deltas": period_report,
            "months_improved": int(
                sum(row["score"] > 0.0 for row in monthly_deltas)
            ),
            "monthly_deltas": monthly_deltas,
            "bootstrap_private_positive_probability": {
                metric: float(np.mean(bootstrap[:, i] > 0.0))
                for i, metric in enumerate(metric_names)
            },
            "bootstrap_private_mean_delta": {
                metric: float(np.mean(bootstrap[:, i]))
                for i, metric in enumerate(metric_names)
            },
            "bootstrap_private_p05_delta": {
                metric: float(np.quantile(bootstrap[:, i], 0.05))
                for i, metric in enumerate(metric_names)
            },
        }

    report["checks"] = {
        "full_all_components_positive": all(
            all(value > 0.0 for value in proxy["period_deltas"]["full"].values())
            for proxy in report["proxies"].values()
        ),
        "h2_all_scores_positive": all(
            proxy["period_deltas"]["h2"]["score"] > 0.0
            for proxy in report["proxies"].values()
        ),
        "h1_all_scores_nonnegative_with_tolerance": all(
            proxy["period_deltas"]["h1"]["score"] >= -1e-6
            for proxy in report["proxies"].values()
        ),
    }
    return report


def make_candidate(
    base_path: Path,
    output_path: Path,
    alpha: float,
    max_delta_ratio: float,
) -> dict[str, float | int | str]:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    missing = [column for column in [*KEY_COLUMNS, *CAPACITY_KWH] if column not in base]
    if missing:
        raise ValueError(f"Base submission is missing columns: {missing}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(base["forecast_kst_dtm"]))
    candidate, mask = smooth_group_3(
        base["kpx_group_1"].to_numpy(dtype=float),
        base["kpx_group_2"].to_numpy(dtype=float),
        base[TARGET].to_numpy(dtype=float),
        timestamps,
        alpha,
        max_delta_ratio,
    )
    delta = candidate - base[TARGET].to_numpy(dtype=float)
    output = base.copy()
    output[TARGET] = candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "base": str(base_path),
        "output": str(output_path),
        "rows": int(len(output)),
        "changed_rows": int(np.count_nonzero(delta)),
        "gate_rows": int(mask.sum()),
        "mean_absolute_movement": float(np.mean(np.abs(delta))),
        "p95_absolute_movement": float(np.quantile(np.abs(delta), 0.95)),
        "max_absolute_movement": float(np.max(np.abs(delta))),
        "groups_1_and_2_unchanged": bool(
            np.array_equal(output["kpx_group_1"], base["kpx_group_1"])
            and np.array_equal(output["kpx_group_2"], base["kpx_group_2"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--base", default="submissions/archive/blend_best_crossg3_25_agree8_delta6.csv"
    )
    parser.add_argument(
        "--output",
        default="submissions/blend_best_crossg3_traj5_consensus.csv",
    )
    parser.add_argument("--artifact-dir", default="artifacts_trajectory_smoothing")
    parser.add_argument(
        "--weighted-cache", default="artifacts_weighted_metric/prediction_cache.npz"
    )
    parser.add_argument("--global-cache", default="artifacts_global/prediction_cache.npz")
    parser.add_argument("--pool-cache", default="artifacts_final_pool/prediction_cache.npz")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-delta-ratio", type=float, default=0.02)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    labels = pd.read_csv(
        Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig"
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    report = validate_policy(
        labels,
        Path(args.weighted_cache),
        {
            "weighted": Path(args.weighted_cache),
            "global": Path(args.global_cache),
            "pool": Path(args.pool_cache),
        },
        args.alpha,
        args.max_delta_ratio,
        args.n_bootstrap,
        args.seed,
    )
    if not all(report["checks"].values()):
        raise RuntimeError(f"Trajectory policy failed validation checks: {report['checks']}")

    report["final"] = make_candidate(
        Path(args.base), Path(args.output), args.alpha, args.max_delta_ratio
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "trajectory_smoothing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"checks": report["checks"], "final": report["final"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
