from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q1_END = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")


@dataclass(frozen=True)
class Policy:
    coverage: float
    alpha: float
    minimum_abs_residual_ratio: float


def _delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _compare(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _delta(before, after),
    }


def build_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    values = ["gefs_u10_spread", "gefs_v10_spread", "gefs_uv10_spread_norm"]
    aggregate = frame.groupby("forecast_kst_dtm")[values].agg(
        ["mean", "max", "min", "std"]
    )
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    center = (
        frame.loc[frame["grid_id"] == 5, ["forecast_kst_dtm"] + values]
        .drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")
        .add_suffix("_center")
    )
    return aggregate.join(center, how="inner").sort_index()


def _calendar_features(index: pd.DatetimeIndex, base: np.ndarray) -> pd.DataFrame:
    hour_angle = 2.0 * np.pi * index.hour.to_numpy() / 24.0
    day_angle = 2.0 * np.pi * index.dayofyear.to_numpy() / 366.0
    return pd.DataFrame(
        {
            "base_ratio": base / CAPACITY,
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "day_sin": np.sin(day_angle),
            "day_cos": np.cos(day_angle),
        },
        index=index,
    )


def _fit_members(
    features: pd.DataFrame,
    residual: np.ndarray,
    train: np.ndarray,
    selection: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    predictions = []
    reports = []
    for seed in seeds:
        model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=1_500,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(
            features.loc[train],
            residual[train],
            eval_set=[(features.loc[selection], residual[selection])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        predictions.append(model.predict(features))
        reports.append({"seed": seed, "best_iteration": int(model.best_iteration_)})
    return np.asarray(predictions), reports


def _select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    residual_prediction: np.ndarray,
    selection: np.ndarray,
) -> tuple[Policy | None, list[dict[str, object]]]:
    absolute_ratio = np.abs(residual_prediction) / CAPACITY
    eligible_proxy = base / CAPACITY >= 0.10
    pool = absolute_ratio[selection & eligible_proxy]
    rows = []
    for coverage in (0.02, 0.05, 0.075, 0.10):
        threshold = float(np.quantile(pool, 1.0 - coverage))
        gate = eligible_proxy & (absolute_ratio >= threshold)
        for alpha in (0.05, 0.10, 0.20, 0.30, 0.50):
            candidate = base.copy()
            candidate[gate] = np.clip(
                base[gate] + alpha * residual_prediction[gate], 0.0, CAPACITY
            )
            comparison = _compare(truth, base, candidate, selection)
            delta = comparison["delta"]
            rows.append(
                {
                    "policy": asdict(Policy(coverage, alpha, threshold)),
                    "selection": comparison,
                    "selection_changed_ratio": float((gate & selection).sum() / selection.sum()),
                    "passes_all_components": min(delta.values()) >= 0.0,
                }
            )
    rows.sort(key=lambda row: row["selection"]["delta"]["score"], reverse=True)
    passing = [row for row in rows if row["passes_all_components"]]
    if not passing:
        return None, rows[:20]
    return Policy(**passing[0]["policy"]), rows[:20]


def _apply_policy(
    base: np.ndarray,
    residual_prediction: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray]:
    gate = (
        (base / CAPACITY >= 0.10)
        & (np.abs(residual_prediction) / CAPACITY >= policy.minimum_abs_residual_ratio)
    )
    candidate = base.copy()
    candidate[gate] = np.clip(
        base[gate] + policy.alpha * residual_prediction[gate], 0.0, CAPACITY
    )
    return candidate, gate


def _screen_family(
    name: str,
    features: pd.DataFrame,
    truth: np.ndarray,
    base: np.ndarray,
    train: np.ndarray,
    selection: np.ndarray,
    locked: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    seed_predictions, fits = _fit_members(
        features, truth - base, train, selection, seeds
    )
    prediction = seed_predictions.mean(axis=0)
    policy, leaderboard = _select_policy(
        truth, base, prediction, selection
    )
    if policy is None:
        return {
            "name": name,
            "feature_count": int(features.shape[1]),
            "fits": fits,
            "selection_status": "rejected_no_all_component_policy",
            "policy": None,
            "selection_leaderboard": leaderboard,
            "locked_h2": None,
            "locked_changed_ratio": None,
            "seed_locked_h2": [],
            "monthly_locked_h2_delta": {},
        }
    candidate, gate = _apply_policy(base, prediction, policy)
    locked_result = _compare(truth, base, candidate, locked)
    seed_locked = []
    for seed, seed_prediction in zip(seeds, seed_predictions):
        seed_candidate, seed_gate = _apply_policy(base, seed_prediction, policy)
        seed_locked.append(
            {
                "seed": seed,
                "changed_ratio": float((seed_gate & locked).sum() / locked.sum()),
                "delta": _compare(truth, base, seed_candidate, locked)["delta"],
            }
        )
    monthly = {
        str(month): _compare(
            truth,
            base,
            candidate,
            locked & (features.index.month == month),
        )["delta"]
        for month in range(7, 13)
    }
    return {
        "name": name,
        "feature_count": int(features.shape[1]),
        "fits": fits,
        "selection_status": "passed",
        "policy": asdict(policy),
        "selection_leaderboard": leaderboard,
        "locked_h2": locked_result,
        "locked_changed_ratio": float((gate & locked).sum() / locked.sum()),
        "seed_locked_h2": seed_locked,
        "monthly_locked_h2_delta": monthly,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="artifacts_final/external_weather/noaa_gefs_train_2023_2024/features.csv",
    )
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_gefs_train_2023_2024/residual_screen.json",
    )
    parser.add_argument("--seeds", default="29101,29102,29103")
    args = parser.parse_args()

    external = build_features(Path(args.features))
    driver = np.load(args.driver_cache)
    meta = np.load(args.meta_cache)
    index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not index.equals(driver_index):
        raise ValueError("driver and meta OOF indexes differ")
    common = index.intersection(external.index)
    common = common[(common >= pd.Timestamp("2024-01-01")) & (common < END)]
    positions = index.get_indexer(common)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)[positions]
    base = meta["valid_candidate"].astype(float)[positions]
    external = external.reindex(common)
    if external.isna().any().any() or not np.isfinite(truth).all():
        raise ValueError("screen inputs contain missing or non-finite values")

    calendar = _calendar_features(common, base)
    with_gefs = calendar.join(external)
    train = np.asarray(common < Q1_END)
    selection = np.asarray((common >= Q1_END) & (common < H2_START))
    locked = np.asarray(common >= H2_START)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    baseline = _screen_family(
        "calendar_base_control",
        calendar,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )
    gefs = _screen_family(
        "calendar_base_plus_gefs_spread",
        with_gefs,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )

    eligible = truth >= 0.10 * CAPACITY
    absolute_error = np.abs(truth - base) / CAPACITY
    diagnostic = {}
    for column in external.columns:
        diagnostic[column] = {
            "spearman_abs_error_eligible": float(
                pd.Series(external[column].to_numpy()[eligible]).corr(
                    pd.Series(absolute_error[eligible]), method="spearman"
                )
            ),
            "spearman_signed_residual_eligible": float(
                pd.Series(external[column].to_numpy()[eligible]).corr(
                    pd.Series((truth - base)[eligible]), method="spearman"
                )
            ),
        }
    report = {
        "family": "noaa_gefs_spread_residual_screen",
        "split": {
            "train": "2024-01-01 through 2024-03-31",
            "selection": "2024-04-01 through 2024-07-01 00:00",
            "locked_h2": "2024-07-01 01:00 through 2024-12-31 23:00",
            "common_rows": int(len(common)),
        },
        "control": baseline,
        "with_gefs": gefs,
        "incremental_locked_h2_vs_control": (
            {
                key: float(
                    gefs["locked_h2"]["delta"][key]
                    - baseline["locked_h2"]["delta"][key]
                )
                for key in ("score", "one_minus_nmae", "ficr")
            }
            if gefs["locked_h2"] is not None and baseline["locked_h2"] is not None
            else None
        ),
        "gefs_correlations": diagnostic,
        "decision": {
            "selection_status": gefs["selection_status"],
            "locked_h2_opened": gefs["locked_h2"] is not None,
            "gefs_locked_components_positive": (
                all(
                    gefs["locked_h2"]["delta"][key] > 0.0
                    for key in ("score", "one_minus_nmae", "ficr")
                )
                if gefs["locked_h2"] is not None
                else False
            ),
            "gefs_incremental_score_vs_control_positive": (
                gefs["locked_h2"] is not None
                and baseline["locked_h2"] is not None
                and gefs["locked_h2"]["delta"]["score"]
                > baseline["locked_h2"]["delta"]["score"]
            ),
            "all_seed_components_positive": bool(gefs["seed_locked_h2"])
            and all(
                min(item["delta"].values()) > 0.0
                for item in gefs["seed_locked_h2"]
            ),
            "all_locked_month_ficr_nonnegative": bool(
                gefs["monthly_locked_h2_delta"]
            )
            and all(
                item["ficr"] >= 0.0
                for item in gefs["monthly_locked_h2_delta"].values()
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
