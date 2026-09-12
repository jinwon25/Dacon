from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.calibration import ShrunkResidualCalibrator
from src.data_audit import prediction_day
from src.diagnostics import slice_diagnostics
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.feature_cache import load_or_build_features
from src.features import TIME_COL
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from src.validation import assign_oof_roles, issue_day_block_bootstrap, make_expanding_year_folds
from train import calibrate, make_model, select_feature_columns


@dataclass(frozen=True)
class Candidate:
    name: str
    train_variant: str
    generation_lambda: float


CANDIDATES = (
    Candidate("e1_all_uniform", "all", 0.0),
    Candidate("e1_eligible_uniform", "eligible_only", 0.0),
    Candidate("e2_eligible_weight025", "eligible_only", 0.25),
    Candidate("e2_eligible_weight050", "eligible_only", 0.50),
    Candidate("e2_eligible_weight075", "eligible_only", 0.75),
    Candidate("e2_eligible_weight100", "eligible_only", 1.00),
)


def _sample_weight(y: pd.Series, capacity: float, generation_lambda: float) -> np.ndarray | None:
    if generation_lambda == 0:
        return None
    normalized_y = np.clip(y.to_numpy(dtype=float) / capacity, 0.0, None)
    return (1.0 - generation_lambda) + generation_lambda * normalized_y


def _fit_fixed_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    candidate: Candidate,
    capacity: float,
    seed: int,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    mask = train_mask & y.notna().to_numpy()
    if candidate.train_variant == "eligible_only":
        mask &= y.to_numpy(dtype=float) >= 0.10 * capacity
    model = make_model(seed=seed, n_estimators=n_estimators)
    model.fit(
        X.loc[mask],
        y.loc[mask],
        sample_weight=_sample_weight(y.loc[mask], capacity, candidate.generation_lambda),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def _representative_weather(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["lead_hour"] = X["lead_hour"]
    speed_col = "gfs__ws100__mean"
    if speed_col in X:
        out["wind_speed"] = X[speed_col]
    u = "gfs__heightAboveGround_100_100u__mean"
    v = "gfs__heightAboveGround_100_100v__mean"
    if u in X and v in X:
        out["wind_direction"] = np.degrees(np.arctan2(X[v], X[u])) % 360.0
    lu = "ldaps__heightAboveGround_10_10u__mean"
    lv = "ldaps__heightAboveGround_10_10v__mean"
    gu = "gfs__heightAboveGround_10_10u__mean"
    gv = "gfs__heightAboveGround_10_10v__mean"
    if all(col in X for col in (lu, lv, gu, gv)):
        out["nwp_disagreement"] = np.sqrt((X[lu] - X[gu]) ** 2 + (X[lv] - X[gv]) ** 2)
    return out


def _metric(y: np.ndarray, pred: np.ndarray, capacity: float) -> dict[str, float | int]:
    return evaluate_group(y, np.clip(pred, 0.0, capacity), capacity).to_dict()


def _choose_candidate(
    target_rows: pd.DataFrame,
    candidate_names: list[str],
    capacity: float,
) -> tuple[str, dict[str, dict[str, float | int]]]:
    selection = target_rows.loc[target_rows["role"] == "selection"]
    if selection.empty:
        raise ValueError("No selection OOF is available")
    metrics = {}
    for name in candidate_names:
        part = selection.loc[selection["candidate"] == name]
        metrics[name] = _metric(part["y_true"].to_numpy(), part["y_pred"].to_numpy(), capacity)
    chosen = max(candidate_names, key=lambda name: (metrics[name]["score"], metrics[name]["ficr"]))
    return chosen, metrics


def _choose_transferred_candidate(
    oof: pd.DataFrame,
    candidate_names: list[str],
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for name in candidate_names:
        group_scores = []
        for target in ("kpx_group_1", "kpx_group_2"):
            capacity = CAPACITY_KWH[target]
            part = oof.loc[(oof["target"] == target) & (oof["candidate"] == name) & (oof["role"] == "selection")]
            group_scores.append(evaluate_group(part["y_true"], part["y_pred"], capacity).score)
        scores[name] = float(np.mean(group_scores))
    return max(candidate_names, key=lambda name: scores[name]), scores


def _affine_from_calibration_rows(
    rows: pd.DataFrame,
    capacity: float,
) -> tuple[float, float, float, dict[str, dict[str, float | int]]]:
    calibration_rows = rows.loc[rows["role"] == "calibration"]
    if calibration_rows.empty:
        return 1.0, 0.0, 0.0, {}
    scale, offset, _ = calibrate(
        calibration_rows["y_true"].to_numpy(),
        calibration_rows["y_pred"].to_numpy(),
        capacity,
    )
    strength_metrics: dict[str, dict[str, float | int]] = {}
    for strength in (0.0, 0.5, 1.0):
        pred = calibration_rows["y_pred"].to_numpy() * (1.0 + strength * (scale - 1.0)) + strength * offset
        strength_metrics[str(strength)] = _metric(calibration_rows["y_true"].to_numpy(), pred, capacity)
    strength = max((0.0, 0.5, 1.0), key=lambda value: strength_metrics[str(value)]["score"])
    return float(scale), float(offset), float(strength), strength_metrics


def _policy_frame(
    oof: pd.DataFrame,
    selected: dict[str, str],
    policy_name: str,
) -> pd.DataFrame:
    parts = []
    for target, candidate in selected.items():
        part = oof.loc[(oof["target"] == target) & (oof["candidate"] == candidate)].copy()
        part["policy"] = policy_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _apply_affine(policy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = []
    settings: dict[str, Any] = {}
    for target, part in policy.groupby("target", sort=False):
        capacity = CAPACITY_KWH[target]
        scale, offset, strength, calibration_metrics = _affine_from_calibration_rows(part, capacity)
        calibrated = part.copy()
        calibrated["y_pred"] = np.clip(
            calibrated["y_pred"] * (1.0 + strength * (scale - 1.0)) + strength * offset,
            0.0,
            capacity,
        )
        calibrated["policy"] = str(part["policy"].iloc[0]) + "__affine"
        output.append(calibrated)
        settings[target] = {"scale": scale, "offset": offset, "strength": strength, "calibration_metrics": calibration_metrics}
    return pd.concat(output, ignore_index=True), settings


def _apply_threshold_calibration(policy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = []
    settings: dict[str, Any] = {}
    for target, part in policy.groupby("target", sort=False):
        capacity = CAPACITY_KWH[target]
        calibration_rows = part["role"] == "calibration"
        calibrated = part.copy()
        if calibration_rows.sum() < 200:
            settings[target] = {"strength": 0.0, "reason": "no independent calibration OOF"}
        else:
            model = ShrunkResidualCalibrator(capacity=capacity, min_samples=150).fit(
                part.loc[calibration_rows, "y_true"].to_numpy(),
                part.loc[calibration_rows, "y_pred"].to_numpy(),
                part.loc[calibration_rows, "lead_hour"].to_numpy(),
            )
            calibrated["y_pred"] = model.predict(part["y_pred"].to_numpy(), part["lead_hour"].to_numpy())
            settings[target] = {
                "strength": model.strength,
                "global_offset": model.global_offset,
                "min_samples": model.min_samples,
                "power_bins": model.power_bins,
                "corrections": {f"{key[0]}:{key[1]}": value for key, value in (model.corrections or {}).items()},
            }
        calibrated["policy"] = str(part["policy"].iloc[0]) + "__threshold"
        output.append(calibrated)
    return pd.concat(output, ignore_index=True), settings


def _evaluate_policy(policy: pd.DataFrame, year: int = 2024) -> dict[str, Any]:
    y_true: dict[str, np.ndarray] = {}
    y_pred: dict[str, np.ndarray] = {}
    groups: dict[str, Any] = {}
    year_mask = prediction_day(policy["timestamp"]).year == year
    for target, part in policy.loc[year_mask].groupby("target", sort=False):
        capacity = CAPACITY_KWH[target]
        y_true[target] = part["y_true"].to_numpy()
        y_pred[target] = part["y_pred"].to_numpy()
        groups[target] = evaluate_group(y_true[target], y_pred[target], capacity).to_dict()
    return {"macro": evaluate_competition(y_true, y_pred), "groups": groups}


def _policy_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro": {key: float(candidate["macro"][key] - baseline["macro"][key]) for key in ("score", "one_minus_nmae", "ficr")},
        "groups": {
            target: {key: float(candidate["groups"][target][key] - baseline["groups"][target][key]) for key in ("score", "one_minus_nmae", "ficr")}
            for target in baseline["groups"]
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("Loading/building leakage-checked baseline features...", flush=True)
    X_all = load_or_build_features(data_dir, "train", artifact_dir / "feature_cache", rebuild=args.rebuild_features)
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_all.index)
    weather = _representative_weather(X_all)

    rows: list[pd.DataFrame] = []
    fold_metrics: dict[str, Any] = {}
    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        columns = select_feature_columns(X_all, target, "base")
        X = X_all[columns]
        folds = make_expanding_year_folds(X.index, y, target)
        fold_metrics[target] = {}
        for fold_i, fold in enumerate(folds, start=1):
            fold_metrics[target][fold.name] = {}
            for candidate_i, candidate in enumerate(CANDIDATES, start=1):
                print(f"FIT {target} {fold.name} {candidate.name}", flush=True)
                seed = args.seed + target_i * 1_000 + fold_i * 100 + candidate_i
                model = _fit_fixed_lgbm(X, y, fold.train, candidate, capacity, seed, args.n_estimators)
                pred = np.clip(model.predict(X.loc[fold.valid]), 0.0, capacity)
                truth = y.loc[fold.valid].to_numpy(dtype=float)
                metric = evaluate_group(truth, pred, capacity).to_dict()
                fold_metrics[target][fold.name][candidate.name] = metric
                part = pd.DataFrame({
                    "timestamp": X.index[fold.valid],
                    "target": target,
                    "fold": fold.name,
                    "candidate": candidate.name,
                    "y_true": truth,
                    "y_pred": pred,
                })
                for col in weather.columns:
                    part[col] = weather.loc[fold.valid, col].to_numpy()
                rows.append(part)
                print(f"METRIC {metric}", flush=True)
    oof = pd.concat(rows, ignore_index=True)
    for target, part_index in oof.groupby("target").groups.items():
        unique_index = pd.DatetimeIndex(sorted(oof.loc[part_index, "timestamp"].unique()))
        role_by_time = dict(zip(unique_index, assign_oof_roles(unique_index)))
        oof.loc[part_index, "role"] = oof.loc[part_index, "timestamp"].map(role_by_time)
    save_oof_predictions(
        oof,
        artifact_dir / "oof_predictions.csv",
        {
            "fold_contract": "expanding complete years, past -> future",
            "selection": "penultimate OOF year H1",
            "calibration": "penultimate OOF year H2",
            "evaluation": "latest OOF year (2024)",
            "candidate_specs": [candidate.__dict__ for candidate in CANDIDATES],
            "n_estimators_fixed": args.n_estimators,
        },
    )

    e1_names = [candidate.name for candidate in CANDIDATES if candidate.name.startswith("e1_")]
    e2_names = [candidate.name for candidate in CANDIDATES if candidate.name.startswith("e2_")]
    selected_e1: dict[str, str] = {}
    selected_e2: dict[str, str] = {}
    selection_metrics: dict[str, Any] = {}
    for target in ("kpx_group_1", "kpx_group_2"):
        target_rows = oof.loc[oof["target"] == target]
        selected_e1[target], e1_metric = _choose_candidate(target_rows, e1_names, CAPACITY_KWH[target])
        selected_e2[target], e2_metric = _choose_candidate(target_rows, e2_names, CAPACITY_KWH[target])
        selection_metrics[target] = {"e1": e1_metric, "e2": e2_metric}
    selected_e1["kpx_group_3"], transfer_e1 = _choose_transferred_candidate(oof, e1_names)
    selected_e2["kpx_group_3"], transfer_e2 = _choose_transferred_candidate(oof, e2_names)
    selection_metrics["group3_transfer"] = {"e1": transfer_e1, "e2": transfer_e2}

    baseline = _policy_frame(oof, selected_e1, "corrected_baseline")
    e1_affine, e1_affine_settings = _apply_affine(baseline)
    weighted = _policy_frame(oof, selected_e2, "generation_weighted")
    weighted_affine, weighted_affine_settings = _apply_affine(weighted)
    weighted_threshold, threshold_settings = _apply_threshold_calibration(weighted)
    policies = {
        "corrected_baseline": baseline,
        "e1_affine": e1_affine,
        "e2_generation_weighted": weighted,
        "e2_generation_weighted_affine": weighted_affine,
        "e3_threshold_calibrated": weighted_threshold,
    }
    evaluations = {name: _evaluate_policy(frame) for name, frame in policies.items()}
    baseline_evaluation = evaluations["corrected_baseline"]
    deltas = {name: _policy_delta(value, baseline_evaluation) for name, value in evaluations.items()}
    bootstraps: dict[str, Any] = {}
    for name, policy in policies.items():
        if name == "corrected_baseline":
            continue
        bootstraps[name] = {}
        for target in CAPACITY_KWH:
            base_year = prediction_day(baseline["timestamp"]).year == 2024
            policy_year = prediction_day(policy["timestamp"]).year == 2024
            base_part = baseline.loc[(baseline["target"] == target) & base_year].sort_values("timestamp")
            candidate_part = policy.loc[(policy["target"] == target) & policy_year].sort_values("timestamp")
            if not base_part["timestamp"].reset_index(drop=True).equals(candidate_part["timestamp"].reset_index(drop=True)):
                raise ValueError(f"Policy timestamps differ for {target}")
            bootstraps[name][target] = issue_day_block_bootstrap(
                base_part["timestamp"],
                base_part["y_true"].to_numpy(),
                base_part["y_pred"].to_numpy(),
                candidate_part["y_pred"].to_numpy(),
                CAPACITY_KWH[target],
                repetitions=args.bootstrap_repetitions,
                seed=args.seed + 50_000 + len(name) + int(target[-1]),
            )

    eligible_policies = []
    for name in policies:
        if name == "corrected_baseline":
            continue
        macro_delta = deltas[name]["macro"]["score"]
        worst_group_delta = min(value["score"] for value in deltas[name]["groups"].values())
        stable = all(value["q05"] >= 0.0 for value in bootstraps[name].values())
        if macro_delta > 0 and worst_group_delta >= 0 and stable:
            eligible_policies.append(name)
    best_name = max(eligible_policies, key=lambda name: evaluations[name]["macro"]["score"]) if eligible_policies else "corrected_baseline"

    diagnostics: dict[str, Any] = {}
    for policy_name in ("corrected_baseline", best_name):
        diagnostics[policy_name] = {}
        policy_year = prediction_day(policies[policy_name]["timestamp"]).year == 2024
        for target, part in policies[policy_name].loc[policy_year].groupby("target"):
            columns = [col for col in ("timestamp", "y_true", "y_pred", "lead_hour", "wind_speed", "wind_direction", "nwp_disagreement") if col in part]
            diagnostics[policy_name][target] = slice_diagnostics(part[columns], CAPACITY_KWH[target])

    runtime = time.perf_counter() - started
    report = {
        "experiment_id": "robust_cv_p2_20260804",
        "official_metric": True,
        "n_estimators_fixed": args.n_estimators,
        "fold_metrics": fold_metrics,
        "selection_metrics": selection_metrics,
        "selected_candidates": {"e1": selected_e1, "e2": selected_e2},
        "calibration": {
            "e1_affine": e1_affine_settings,
            "e2_affine": weighted_affine_settings,
            "e3_threshold": threshold_settings,
        },
        "evaluations_2024": evaluations,
        "deltas_vs_corrected_baseline": deltas,
        "day_block_bootstrap": bootstraps,
        "promotion_eligible": eligible_policies,
        "selected_best": best_name,
        "diagnostics": diagnostics,
        "runtime_seconds": runtime,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in policies:
        delta = deltas[name]
        bootstrap = bootstraps.get(name, {})
        keep = name == "corrected_baseline" or name in eligible_policies
        append_experiment_result(
            args.results_csv,
            {
                "experiment_id": f"robust_cv_p2_20260804__{name}",
                "hypothesis": {
                    "corrected_baseline": "official metric changes all/eligible selection",
                    "e1_affine": "independent H2 affine calibration improves the locked E1 model",
                    "e2_generation_weighted": "actual-normalized generation weights improve official FiCR",
                    "e2_generation_weighted_affine": "affine calibration complements generation weighting",
                    "e3_threshold_calibrated": "prediction-bin and lead-bin residual shrinkage improves threshold accuracy",
                }[name],
                "feature_set": ["base", "actual_lead_from_availability"],
                "model": "LightGBM L1",
                "hyperparameters": {"n_estimators": args.n_estimators, "candidates": selected_e1 if name.startswith(("corrected", "e1")) else selected_e2},
                "fold_scores": evaluations[name],
                "group_metrics": evaluations[name]["groups"],
                "mean_delta": delta["macro"],
                "worst_fold_delta": min((value["score"] for value in delta["groups"].values()), default=0.0),
                "bootstrap_interval": bootstrap,
                "runtime_seconds": runtime,
                "decision": "keep" if keep else "reject",
                "reason": "baseline" if name == "corrected_baseline" else ("positive in every group with non-negative day-block q05" if keep else "failed mean/group/bootstrap promotion gate"),
            },
        )
    print(json.dumps({"selected_best": best_name, "evaluations_2024": evaluations, "deltas": deltas, "runtime_seconds": runtime}, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe P2 year-forward experiment suite.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
