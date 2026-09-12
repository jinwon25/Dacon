"""P5: leakage-safe NWP -> site wind -> monotone power-curve experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiments.postprocess_robust_cv import macro_issue_day_bootstrap
from experiments.robust_cv_suite import _evaluate_policy, _policy_delta, _policy_frame
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_group
from src.scada import load_hourly_scada
from src.validation import issue_day_block_bootstrap


def _site_columns(frame: pd.DataFrame, target: str) -> list[str]:
    signals = ("__ws", "__hub_u", "__hub_v", "__hub_dir", "__u", "__v")
    own = [col for col in frame.columns if target in col and any(signal in col for signal in signals)]
    time = [col for col in ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_hour") if col in frame]
    if not own:
        raise ValueError(f"No site-aware NWP columns found for {target}")
    return sorted(set(own + time))


def _consensus_hub_wind(frame: pd.DataFrame, target: str) -> pd.Series:
    cols = [
        f"ldaps__{target}__hub_ws117__idw",
        f"gfs__{target}__hub_ws117__idw",
    ]
    missing = [col for col in cols if col not in frame]
    if missing:
        raise ValueError(f"Missing hub-wind columns: {missing}")
    return frame[cols].mean(axis=1)


def _site_wind_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=250,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def _fit_physical_candidates(
    X_train: pd.DataFrame,
    scada_train: pd.DataFrame,
    X_query: pd.DataFrame,
    target: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capacity = CAPACITY_KWH[target]
    clean = scada_train["is_clean_for_curve"].eq(True)
    clean &= scada_train["scada_power_kwh"].notna() & scada_train["scada_wind_speed"].notna()
    clean &= X_train.notna().any(axis=1)
    if clean.sum() < 1_000:
        raise ValueError(f"Insufficient clean SCADA rows for {target}: {clean.sum()}")

    curve = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    curve.fit(
        scada_train.loc[clean, "scada_wind_speed"].to_numpy(),
        np.clip(scada_train.loc[clean, "scada_power_kwh"].to_numpy() / capacity, 0.0, 1.0),
    )

    site_cols = _site_columns(X_train, target)
    wind_model = _site_wind_model(seed)
    wind_model.fit(
        X_train.loc[clean, site_cols],
        scada_train.loc[clean, "scada_wind_speed"],
        callbacks=[lgb.log_evaluation(0)],
    )
    predicted_site_wind = np.clip(wind_model.predict(X_query[site_cols]), 0.0, 60.0)
    two_stage = np.clip(curve.predict(predicted_site_wind) * capacity, 0.0, capacity)

    direct_curve = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    nwp_train_wind = _consensus_hub_wind(X_train, target)
    direct_mask = clean & nwp_train_wind.notna()
    direct_curve.fit(
        nwp_train_wind.loc[direct_mask].to_numpy(),
        np.clip(scada_train.loc[direct_mask, "scada_power_kwh"].to_numpy() / capacity, 0.0, 1.0),
    )
    direct = np.clip(direct_curve.predict(_consensus_hub_wind(X_query, target).to_numpy()) * capacity, 0.0, capacity)
    return {
        "isotonic_nwp": direct,
        "two_stage_isotonic": two_stage,
    }, {
        "clean_curve_rows": int(clean.sum()),
        "site_feature_count": len(site_cols),
        "site_wind_train_mae": float(np.mean(np.abs(
            wind_model.predict(X_train.loc[clean, site_cols]) - scada_train.loc[clean, "scada_wind_speed"].to_numpy()
        ))),
    }


def _best_alpha(y_true: np.ndarray, direct: np.ndarray, physical: np.ndarray, capacity: float) -> tuple[float, dict[str, Any]]:
    grid = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
    scores: dict[str, Any] = {}
    best_alpha = 0.0
    best_score = -np.inf
    for alpha in grid:
        pred = np.clip((1.0 - alpha) * direct + alpha * physical, 0.0, capacity)
        metric = evaluate_group(y_true, pred, capacity).to_dict()
        scores[str(alpha)] = metric
        if metric["score"] > best_score:
            best_score = float(metric["score"])
            best_alpha = alpha
    return best_alpha, scores


def _year(frame: pd.DataFrame | pd.DatetimeIndex) -> pd.Series:
    index = frame.index if isinstance(frame, pd.DataFrame) else frame
    return pd.Series(prediction_day(index).year, index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache", default="artifacts/robust_cv_p2/feature_cache/features_train.pkl")
    parser.add_argument("--p2-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--artifact-dir", default="artifacts/scada_physical")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_pickle(args.feature_cache)
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm").reindex(X.index)
    scada = {target: frame.reindex(X.index) for target, frame in load_hourly_scada(args.data_dir).items()}
    p2_report = json.loads((Path(args.p2_dir) / "report.json").read_text(encoding="utf-8"))
    p2_oof = pd.read_csv(Path(args.p2_dir) / "oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    raw_direct = _policy_frame(p2_oof, p2_report["selected_candidates"]["e1"], "raw_direct")
    safe = pd.read_csv(Path(args.p2_dir) / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    safe = safe.loc[safe["policy"] == "safe_cv_best"].copy()

    physical_rows: list[pd.DataFrame] = []
    training_reports: dict[str, Any] = {}
    selected_methods: dict[str, str] = {}
    alphas: dict[str, float] = {}
    alpha_grids: dict[str, Any] = {}
    forecast_year = _year(X)

    # Group 1/2 provide a clean prior-year selection/calibration split.
    for group_i, target in enumerate(("kpx_group_1", "kpx_group_2"), start=1):
        train_2022 = forecast_year < 2023
        valid_2023 = (forecast_year == 2023) & labels[target].notna()
        candidates, fit_report = _fit_physical_candidates(
            X.loc[train_2022], scada[target].loc[train_2022], X.loc[valid_2023], target, args.seed + group_i,
        )
        truth = labels.loc[valid_2023, target]
        base = raw_direct.loc[(raw_direct["target"] == target) & (prediction_day(raw_direct["timestamp"]).year == 2023)].set_index("timestamp").reindex(truth.index)
        selection = prediction_day(truth.index).month <= 6
        selection_metrics = {
            name: evaluate_group(truth.loc[selection], pred[selection], CAPACITY_KWH[target]).to_dict()
            for name, pred in candidates.items()
        }
        selected = max(selection_metrics, key=lambda name: selection_metrics[name]["score"])
        calibration = ~selection
        alpha, grid = _best_alpha(
            truth.loc[calibration].to_numpy(),
            base.loc[calibration, "y_pred"].to_numpy(),
            candidates[selected][calibration],
            CAPACITY_KWH[target],
        )
        selected_methods[target] = selected
        alphas[target] = alpha
        alpha_grids[target] = grid
        training_reports[target] = {"fit_2022": fit_report, "selection_2023_h1": selection_metrics}
        for name, pred in candidates.items():
            physical_rows.append(pd.DataFrame({
                "timestamp": truth.index, "target": target, "fold": "2022_to_2023", "candidate": name,
                "y_true": truth.to_numpy(), "y_pred": pred,
            }))

    # Fixed transfer prevents using group-3's evaluation year for its method/weight.
    method_score = {
        name: float(np.mean([
            training_reports[target]["selection_2023_h1"][name]["score"]
            for target in ("kpx_group_1", "kpx_group_2")
        ]))
        for name in ("isotonic_nwp", "two_stage_isotonic")
    }
    transferred_method = max(method_score, key=method_score.get)
    transferred_alpha = float(np.median([alphas["kpx_group_1"], alphas["kpx_group_2"]]))
    selected_methods["kpx_group_3"] = transferred_method
    alphas["kpx_group_3"] = transferred_alpha

    # Refit strictly on years before the locked 2024 evaluation.
    for group_i, target in enumerate(CAPACITY_KWH, start=1):
        train_prior = forecast_year < 2024
        valid_2024 = (forecast_year == 2024) & labels[target].notna()
        candidates, fit_report = _fit_physical_candidates(
            X.loc[train_prior], scada[target].loc[train_prior], X.loc[valid_2024], target, args.seed + 100 + group_i,
        )
        truth = labels.loc[valid_2024, target]
        training_reports.setdefault(target, {})["fit_prior_to_2024"] = fit_report
        for name, pred in candidates.items():
            physical_rows.append(pd.DataFrame({
                "timestamp": truth.index, "target": target, "fold": "prior_years_to_2024", "candidate": name,
                "y_true": truth.to_numpy(), "y_pred": pred,
            }))

    physical = pd.concat(physical_rows, ignore_index=True)
    save_oof_predictions(physical, artifact_dir / "oof_predictions.csv", {"SCADA_usage": "training targets only; never inference input"})

    candidate_parts = []
    complementarity: dict[str, Any] = {}
    for target, capacity in CAPACITY_KWH.items():
        safe_part = safe.loc[(safe["target"] == target) & (prediction_day(safe["timestamp"]).year == 2024)].sort_values("timestamp").copy()
        physical_part = physical.loc[
            (physical["target"] == target)
            & (physical["fold"] == "prior_years_to_2024")
            & (physical["candidate"] == selected_methods[target])
        ].sort_values("timestamp")
        if not safe_part["timestamp"].reset_index(drop=True).equals(physical_part["timestamp"].reset_index(drop=True)):
            raise ValueError(f"Physical and direct timestamps differ for {target}")
        physical_pred = physical_part["y_pred"].to_numpy()
        alpha = alphas[target]
        safe_part["y_pred"] = np.clip((1.0 - alpha) * safe_part["y_pred"].to_numpy() + alpha * physical_pred, 0.0, capacity)
        safe_part["policy"] = "scada_physical_blend"
        candidate_parts.append(safe_part)
        eligible = safe_part["y_true"].to_numpy() >= 0.10 * capacity
        direct_residual = safe.loc[
            (safe["target"] == target) & (prediction_day(safe["timestamp"]).year == 2024)
        ].sort_values("timestamp")["y_pred"].to_numpy() - safe_part["y_true"].to_numpy()
        physical_residual = physical_pred - safe_part["y_true"].to_numpy()
        complementarity[target] = {
            "selected_method": selected_methods[target],
            "blend_alpha": alpha,
            "eligible_residual_correlation": float(np.corrcoef(direct_residual[eligible], physical_residual[eligible])[0, 1]),
            "eligible_prediction_correlation": float(np.corrcoef(
                direct_residual[eligible] + safe_part.loc[eligible, "y_true"].to_numpy(), physical_pred[eligible]
            )[0, 1]),
        }
    candidate = pd.concat(candidate_parts, ignore_index=True)
    baseline_eval = _evaluate_policy(safe)
    candidate_eval = _evaluate_policy(candidate)
    delta = _policy_delta(candidate_eval, baseline_eval)
    group_bootstrap = {}
    for group_i, target in enumerate(CAPACITY_KWH, start=1):
        base_part = safe.loc[(safe["target"] == target) & (prediction_day(safe["timestamp"]).year == 2024)].sort_values("timestamp")
        candidate_part = candidate.loc[candidate["target"] == target].sort_values("timestamp")
        group_bootstrap[target] = issue_day_block_bootstrap(
            base_part["timestamp"], base_part["y_true"], base_part["y_pred"], candidate_part["y_pred"],
            CAPACITY_KWH[target], repetitions=args.bootstrap_repetitions, seed=args.seed + group_i,
        )
    macro_bootstrap = macro_issue_day_bootstrap(safe, candidate, args.bootstrap_repetitions, args.seed + 50)
    promoted = macro_bootstrap["q05"] >= 0.0 and all(
        group_bootstrap[target]["q05"] >= 0.0 for target in CAPACITY_KWH if alphas[target] > 0
    )
    runtime = time.perf_counter() - started
    report = {
        "experiment_id": "p5_scada_physical_20260804",
        "SCADA_contract": "SCADA is used only as a training target; no test-period SCADA is read or assumed",
        "methods": ["isotonic_nwp", "two_stage_isotonic"],
        "training": training_reports,
        "transferred_group3_selection": {"method_scores_g12": method_score, "method": transferred_method, "alpha": transferred_alpha},
        "alpha_grids_2023_h2": alpha_grids,
        "complementarity_2024": complementarity,
        "safe_baseline_2024": baseline_eval,
        "candidate_2024": candidate_eval,
        "delta_vs_safe": delta,
        "bootstrap": {"groups": group_bootstrap, "macro": macro_bootstrap},
        "promotion_eligible": promoted,
        "runtime_seconds": runtime,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(candidate, artifact_dir / "composite_oof_predictions.csv", {"policy": "SCADA physical blend"})
    append_experiment_result(args.results_csv, {
        "experiment_id": report["experiment_id"],
        "hypothesis": "training-only SCADA site-wind and monotone power curve adds complementary physical signal",
        "feature_set": ["site NWP", "training SCADA wind/power", "monotone power curve"],
        "model": "LightGBM site-wind -> isotonic power curve; OOF blend",
        "hyperparameters": {"methods": report["methods"], "alphas": alphas},
        "fold_scores": candidate_eval,
        "group_metrics": candidate_eval["groups"],
        "mean_delta": delta["macro"],
        "worst_fold_delta": min(value["score"] for value in delta["groups"].values()),
        "bootstrap_interval": report["bootstrap"],
        "runtime_seconds": runtime,
        "decision": "keep" if promoted else "reject",
        "reason": "stable day-block improvement" if promoted else "OOF blend improvement was absent or unstable",
    })
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
