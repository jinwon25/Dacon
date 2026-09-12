"""SPRINT-066 M1: eligible quantiles and official-utility Bayes actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.postprocess_robust_cv import macro_issue_day_bootstrap
from experiments.robust_cv_suite import (
    CANDIDATES,
    _apply_affine,
    _evaluate_policy,
    _fit_fixed_lgbm,
    _policy_delta,
    _policy_frame,
)
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_group
from src.probabilistic import (
    DEFAULT_QUANTILE_LEVELS,
    enforce_noncrossing,
    metric_aware_bayes_action,
    shrink_action,
)
from src.validation import issue_day_block_bootstrap
from train import select_feature_columns


ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _quantile_model(alpha: float, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def fit_quantile_distribution(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    query_mask: np.ndarray,
    *,
    capacity: float,
    levels: np.ndarray,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible = train_mask & y.notna().to_numpy() & (y.to_numpy(dtype=float) >= 0.10 * capacity)
    if eligible.sum() < 500:
        raise ValueError(f"Insufficient eligible quantile rows: {eligible.sum()}")
    prediction = []
    for quantile_i, level in enumerate(levels, start=1):
        model = _quantile_model(float(level), seed + quantile_i, n_estimators)
        model.fit(X.loc[eligible], y.loc[eligible], callbacks=[lgb.log_evaluation(0)])
        prediction.append(np.clip(model.predict(X.loc[query_mask]), 0.0, capacity))
    raw = np.column_stack(prediction)
    crossing_rows = int(np.any(np.diff(raw, axis=1) < 0.0, axis=1).sum())
    repaired = enforce_noncrossing(raw)
    return repaired, {
        "eligible_train_rows": int(eligible.sum()),
        "query_rows": int(query_mask.sum()),
        "crossing_rows_before_sort": crossing_rows,
        "crossing_rate_before_sort": float(crossing_rows / max(len(raw), 1)),
        "mean_eligible_generation": float(y.loc[eligible].mean()),
    }


def _metrics(y: np.ndarray, prediction: np.ndarray, capacity: float) -> dict[str, Any]:
    return evaluate_group(y, np.clip(prediction, 0.0, capacity), capacity).to_dict()


def _alpha_table(y: np.ndarray, p50: np.ndarray, action: np.ndarray, capacity: float, mask: np.ndarray) -> dict[str, Any]:
    return {
        str(alpha): _metrics(y[mask], shrink_action(p50[mask], action[mask], alpha, capacity), capacity)
        for alpha in ALPHAS
    }


def _select_alpha(table: dict[str, Any]) -> float:
    return max(ALPHAS, key=lambda alpha: (table[str(alpha)]["score"], table[str(alpha)]["ficr"], -alpha))


def _policy(parts: list[pd.DataFrame], name: str, prediction_col: str) -> pd.DataFrame:
    output = []
    for part in parts:
        frame = part[["timestamp", "target", "fold", "y_true", prediction_col]].rename(columns={prediction_col: "y_pred"}).copy()
        frame["policy"] = name
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def _parse_levels(value: str) -> np.ndarray:
    levels = np.asarray([float(item) for item in value.split(",") if item.strip()], dtype=float)
    if len(levels) < 3 or not np.any(np.isclose(levels, 0.5)):
        raise ValueError("quantile grid must include 0.5 and at least three levels")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache", default="artifacts/robust_cv_p2/feature_cache/features_train.pkl")
    parser.add_argument("--p2-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--artifact-dir", default="artifacts/metric_aware_bayes")
    parser.add_argument("--results-csv", default="artifacts/experiments_066.csv")
    parser.add_argument("--levels", default=",".join(str(value) for value in DEFAULT_QUANTILE_LEVELS))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    levels = _parse_levels(args.levels)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X_all = pd.read_pickle(args.feature_cache)
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm").reindex(X_all.index)
    forecast_day = prediction_day(X_all.index)
    forecast_year = forecast_day.year
    p2_report = json.loads((Path(args.p2_dir) / "report.json").read_text(encoding="utf-8"))
    p2_oof = pd.read_csv(Path(args.p2_dir) / "oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    raw_policy = _policy_frame(p2_oof, p2_report["selected_candidates"]["e1"], "raw_l1")
    affine_policy, affine_settings = _apply_affine(raw_policy)
    e1_eligible = next(candidate for candidate in CANDIDATES if candidate.name == "e1_eligible_uniform")

    locked_parts: list[pd.DataFrame] = []
    group_reports: dict[str, Any] = {}
    selected_alphas: dict[str, float] = {}
    quantile_frames: list[pd.DataFrame] = []
    for group_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        columns = select_feature_columns(X_all, target, "base")
        X = X_all[columns]
        y = labels[target]
        if target in ("kpx_group_1", "kpx_group_2"):
            development_train = forecast_year < 2023
            development_query = (forecast_year == 2023) & y.notna().to_numpy()
            raw_development = raw_policy.loc[(raw_policy["target"] == target) & (prediction_day(raw_policy["timestamp"]).year == 2023)].set_index("timestamp").reindex(X.index[development_query])["y_pred"].to_numpy()
            selection_mask = prediction_day(X.index[development_query]).month <= 6
            confirmation_mask = ~selection_mask
            development_fold = "2022_to_2023"
        else:
            in_2023 = forecast_year == 2023
            development_train = in_2023 & (forecast_day.month <= 6)
            development_query = in_2023 & (forecast_day.month > 6) & y.notna().to_numpy()
            raw_model = _fit_fixed_lgbm(X, y, development_train, e1_eligible, capacity, args.seed + 50, args.n_estimators)
            raw_development = np.clip(raw_model.predict(X.loc[development_query]), 0.0, capacity)
            selection_mask = np.ones(int(development_query.sum()), dtype=bool)
            confirmation_mask = selection_mask
            development_fold = "2023_h1_to_h2"

        print(f"[{target}] fitting development quantiles ({len(levels)} models)", flush=True)
        q_dev, dev_fit = fit_quantile_distribution(
            X, y, development_train, development_query, capacity=capacity, levels=levels,
            n_estimators=args.n_estimators, seed=args.seed + group_i * 1_000,
        )
        p50_i = int(np.argmin(np.abs(levels - 0.5)))
        p50_dev = q_dev[:, p50_i]
        action_dev = metric_aware_bayes_action(
            q_dev,
            capacity=capacity,
            mean_eligible_generation=dev_fit["mean_eligible_generation"],
            levels=levels,
            point_candidates=raw_development,
            reference=p50_dev,
        )
        truth_dev = y.loc[development_query].to_numpy(dtype=float)
        selection_table = _alpha_table(truth_dev, p50_dev, action_dev.action, capacity, selection_mask)
        selected_alpha = _select_alpha(selection_table)
        confirmation_table = _alpha_table(truth_dev, p50_dev, action_dev.action, capacity, confirmation_mask)
        selected_alphas[target] = selected_alpha

        locked_train = (forecast_year < 2024) & y.notna().to_numpy()
        locked_query = (forecast_year == 2024) & y.notna().to_numpy()
        print(f"[{target}] fitting locked 2024 quantiles ({len(levels)} models)", flush=True)
        q_locked, locked_fit = fit_quantile_distribution(
            X, y, locked_train, locked_query, capacity=capacity, levels=levels,
            n_estimators=args.n_estimators, seed=args.seed + group_i * 1_000 + 100,
        )
        p50_locked = q_locked[:, p50_i]
        raw_locked_frame = raw_policy.loc[(raw_policy["target"] == target) & (prediction_day(raw_policy["timestamp"]).year == 2024)].set_index("timestamp").reindex(X.index[locked_query])
        if raw_locked_frame["y_pred"].isna().any():
            raise ValueError(f"Missing raw L1 OOF alignment for {target}")
        raw_locked = raw_locked_frame["y_pred"].to_numpy()
        action_locked = metric_aware_bayes_action(
            q_locked,
            capacity=capacity,
            mean_eligible_generation=locked_fit["mean_eligible_generation"],
            levels=levels,
            point_candidates=raw_locked,
            reference=p50_locked,
        )
        selected_locked = shrink_action(p50_locked, action_locked.action, selected_alpha, capacity)
        timestamps = X.index[locked_query]
        truth_locked = y.loc[locked_query].to_numpy(dtype=float)
        part = pd.DataFrame({
            "timestamp": timestamps,
            "target": target,
            "fold": "prior_years_to_2024",
            "y_true": truth_locked,
            "raw_l1": raw_locked,
            "quantile_p50": p50_locked,
            "bayes_action": action_locked.action,
            "selected_shrinkage": selected_locked,
            "interval_width_90": q_locked[:, -1] - q_locked[:, 0],
            "expected_advantage": action_locked.expected_advantage,
        })
        locked_parts.append(part)
        qframe = part[["timestamp", "target", "fold", "y_true"]].copy()
        for quantile_i, level in enumerate(levels):
            qframe[f"q{level:.2f}"] = q_locked[:, quantile_i]
        quantile_frames.append(qframe)
        group_reports[target] = {
            "features": len(columns),
            "development_fold": development_fold,
            "development_fit": dev_fit,
            "selection_rows": int(selection_mask.sum()),
            "confirmation_rows": int(confirmation_mask.sum()),
            "selection_alpha_metrics": selection_table,
            "selected_alpha": selected_alpha,
            "confirmation_alpha_metrics": confirmation_table,
            "locked_fit": locked_fit,
            "locked_metrics": {
                "raw_l1": _metrics(truth_locked, raw_locked, capacity),
                "quantile_p50": _metrics(truth_locked, p50_locked, capacity),
                "bayes_action": _metrics(truth_locked, action_locked.action, capacity),
                "selected_shrinkage": _metrics(truth_locked, selected_locked, capacity),
            },
            "action": {
                "candidate_count": action_locked.candidate_count,
                "changed_fraction": float(np.mean(np.abs(action_locked.action - p50_locked) > 1e-8)),
                "mean_absolute_movement_ratio": float(np.mean(np.abs(action_locked.action - p50_locked)) / capacity),
                "p95_movement_ratio": float(np.quantile(np.abs(action_locked.action - p50_locked), 0.95) / capacity),
            },
        }

    policies = {
        "raw_l1": _policy(locked_parts, "raw_l1", "raw_l1"),
        "quantile_p50": _policy(locked_parts, "quantile_p50", "quantile_p50"),
        "bayes_action": _policy(locked_parts, "bayes_action", "bayes_action"),
        "selected_shrinkage": _policy(locked_parts, "selected_shrinkage", "selected_shrinkage"),
    }
    evaluations = {name: _evaluate_policy(policy) for name, policy in policies.items()}
    evaluations["global_affine"] = _evaluate_policy(affine_policy)
    deltas_vs_raw = {name: _policy_delta(evaluation, evaluations["raw_l1"]) for name, evaluation in evaluations.items() if name != "raw_l1"}
    deltas_vs_p50 = {name: _policy_delta(evaluation, evaluations["quantile_p50"]) for name, evaluation in evaluations.items() if name != "quantile_p50"}
    selected_policy = policies["selected_shrinkage"]
    group_bootstrap = {}
    for group_i, target in enumerate(CAPACITY_KWH, start=1):
        base = policies["raw_l1"].loc[policies["raw_l1"]["target"] == target].sort_values("timestamp")
        contender = selected_policy.loc[selected_policy["target"] == target].sort_values("timestamp")
        group_bootstrap[target] = issue_day_block_bootstrap(
            base["timestamp"], base["y_true"], base["y_pred"], contender["y_pred"], CAPACITY_KWH[target],
            repetitions=args.bootstrap_repetitions, seed=args.seed + group_i,
        )
    macro_bootstrap = macro_issue_day_bootstrap(
        policies["raw_l1"], selected_policy, args.bootstrap_repetitions, args.seed + 100,
    )
    delta = deltas_vs_raw["selected_shrinkage"]["macro"]
    promotion = (
        delta["score"] >= 0.0015
        and delta["one_minus_nmae"] >= -0.002
        and macro_bootstrap["positive_fraction"] >= 0.80
        and macro_bootstrap["q05"] >= -0.002
    )
    runtime = time.perf_counter() - started
    report = {
        "experiment_id": "s066_m1_eligible_quantile_bayes",
        "formula": "argmax E[-abs(p-Y) + C/(4*mean_y_train_eligible)*Y*price(abs(p-Y)) | Y>=0.1C]",
        "levels": levels.tolist(),
        "validation_contract": {
            "groups_1_2": "2022 fit -> 2023 H1 alpha selection/H2 confirmation; 2022-2023 refit -> locked 2024",
            "group_3": "2023 H1 fit -> H2 alpha selection; full 2023 refit -> locked 2024",
            "quantile_calibration": "row sort only; no locked-year calibration",
            "test_data_used": False,
        },
        "groups": group_reports,
        "selected_alphas": selected_alphas,
        "evaluations_2024": evaluations,
        "deltas_vs_raw_l1": deltas_vs_raw,
        "deltas_vs_quantile_p50": deltas_vs_p50,
        "bootstrap_vs_raw_l1": {"groups": group_bootstrap, "macro": macro_bootstrap},
        "promotion_eligible": promotion,
        "runtime_seconds": runtime,
        "smoke": args.smoke,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(pd.concat(policies.values(), ignore_index=True), artifact_dir / "oof_predictions.csv", report["validation_contract"])
    pd.concat(quantile_frames, ignore_index=True).to_csv(artifact_dir / "quantile_oof_predictions.csv", index=False, encoding="utf-8-sig")
    if not args.smoke:
        append_experiment_result(args.results_csv, {
            "experiment_id": report["experiment_id"],
            "hypothesis": "eligible conditional quantiles support an official-utility Bayes point action that transfers forward",
            "feature_set": ["base", "11 eligible-only conditional quantiles"],
            "model": "LightGBM quantile distribution + metric-aware Bayes action",
            "hyperparameters": {"levels": levels.tolist(), "n_estimators": args.n_estimators, "selected_alphas": selected_alphas},
            "fold_scores": {target: value["locked_metrics"] for target, value in group_reports.items()},
            "group_metrics": evaluations["selected_shrinkage"]["groups"],
            "mean_delta": delta,
            "worst_fold_delta": min(value["score"] for value in deltas_vs_raw["selected_shrinkage"]["groups"].values()),
            "bootstrap_interval": report["bootstrap_vs_raw_l1"],
            "runtime_seconds": runtime,
            "decision": "keep" if promotion else "reject",
            "reason": "passes forward score/NMAE/bootstrap gate" if promotion else "does not beat the corrected raw L1 surface robustly",
        })
    print(json.dumps({
        "selected_alphas": selected_alphas,
        "evaluations_2024": evaluations,
        "deltas_vs_raw_l1": deltas_vs_raw,
        "bootstrap_vs_raw_l1": report["bootstrap_vs_raw_l1"],
        "promotion_eligible": promotion,
        "runtime_seconds": runtime,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
