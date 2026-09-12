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
from experiments.robust_cv_suite import _affine_from_calibration_rows, _evaluate_policy, _policy_delta
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_group
from src.validation import assign_oof_roles, issue_day_block_bootstrap
from train import make_model, select_feature_columns


GROUP_META = {
    "kpx_group_1": {"group_id": 1.0, "capacity": 21_600.0, "manufacturer": 0.0, "turbines": 6.0, "latitude": 37.2871, "longitude": 128.9538},
    "kpx_group_2": {"group_id": 2.0, "capacity": 21_600.0, "manufacturer": 0.0, "turbines": 6.0, "latitude": 37.2823, "longitude": 128.9652},
    "kpx_group_3": {"group_id": 3.0, "capacity": 21_000.0, "manufacturer": 1.0, "turbines": 5.0, "latitude": 37.2752, "longitude": 128.9695},
}


def _group_frame(X: pd.DataFrame, target: str, site_aware: bool) -> pd.DataFrame:
    columns = select_feature_columns(X, target, "base")
    frame = X[columns].copy()
    if site_aware:
        own = [col for col in X if f"__{target}__" in col and "hub_" not in col]
        own_frame = X[own].copy()
        own_frame.columns = [col.replace(f"__{target}__", "__own_group__") for col in own]
        frame = frame.join(own_frame)
    for key, value in GROUP_META[target].items():
        frame[f"group_meta__{key}"] = np.float32(value)
    return frame.astype("float32")


def _stack(
    group_frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    forecast_year: int,
    train: bool,
    eligible_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_parts = []
    y_parts = []
    meta_parts = []
    for target, frame in group_frames.items():
        capacity = CAPACITY_KWH[target]
        y = labels[target].reindex(frame.index)
        years = prediction_day(frame.index).year
        mask = (years < forecast_year) if train else (years == forecast_year)
        mask &= y.notna().to_numpy()
        if train and eligible_only:
            mask &= y.to_numpy(dtype=float) >= 0.10 * capacity
        X_parts.append(frame.loc[mask].reset_index(drop=True))
        y_parts.append(pd.Series(y.loc[mask].to_numpy(dtype=float) / capacity))
        meta_parts.append(pd.DataFrame({"timestamp": frame.index[mask], "target": target, "y_true": y.loc[mask].to_numpy(dtype=float)}))
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True), pd.concat(meta_parts, ignore_index=True)


def _fit_pooled(X: pd.DataFrame, y: pd.Series, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    model = make_model(seed, n_estimators=n_estimators)
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    return model


def _compose_with_safe(candidate_group3: pd.DataFrame, safe: pd.DataFrame, name: str) -> pd.DataFrame:
    parts = [safe.loc[safe["target"].isin(["kpx_group_1", "kpx_group_2"])].copy(), candidate_group3.copy()]
    output = pd.concat(parts, ignore_index=True)
    output["policy"] = name
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 pooled normalized-target group-3 transfer.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache", default="artifacts/robust_cv_p2/feature_cache/features_train.pkl")
    parser.add_argument("--p2-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--artifact-dir", default="artifacts/pooled_group3")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    X = pd.read_pickle(args.feature_cache)
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    safe_all = pd.read_csv(Path(args.p2_dir) / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    safe = safe_all.loc[safe_all["policy"] == "safe_cv_best"].copy()
    safe_evaluation = _evaluate_policy(safe)

    candidate_rows = []
    candidate_reports: dict[str, Any] = {}
    # Explicit tuple avoids a nested-product typo and keeps the two hypotheses ordered.
    for candidate_i, (name, site_aware) in enumerate((("pooled_shared", False), ("pooled_site_aware", True)), start=1):
        frames = {target: _group_frame(X, target, site_aware) for target in CAPACITY_KWH}
        candidate_reports[name] = {"site_aware": site_aware, "features": int(next(iter(frames.values())).shape[1]), "folds": {}}
        for fold_i, year in enumerate((2023, 2024), start=1):
            X_train, y_train, _ = _stack(frames, labels, year, train=True)
            X_valid, _, meta = _stack(frames, labels, year, train=False, eligible_only=False)
            print(f"FIT {name} prior years -> {year} train={len(X_train)} valid={len(X_valid)}", flush=True)
            model = _fit_pooled(X_train, y_train, args.seed + candidate_i * 1_000 + fold_i, args.n_estimators)
            normalized = np.clip(model.predict(X_valid), 0.0, 1.0)
            meta = meta.reset_index(drop=True)
            meta["fold"] = f"prior_years_to_{year}"
            meta["candidate"] = name
            meta["y_pred"] = normalized * meta["target"].map(CAPACITY_KWH).to_numpy(dtype=float)
            meta["lead_hour"] = X_valid["lead_hour"].to_numpy()
            candidate_rows.append(meta)
            candidate_reports[name]["folds"][str(year)] = {
                target: evaluate_group(part["y_true"], part["y_pred"], CAPACITY_KWH[target]).to_dict()
                for target, part in meta.groupby("target")
            }
    oof = pd.concat(candidate_rows, ignore_index=True)
    for candidate, indices in oof.groupby("candidate").groups.items():
        unique_index = pd.DatetimeIndex(sorted(oof.loc[indices, "timestamp"].unique()))
        roles = dict(zip(unique_index, assign_oof_roles(unique_index)))
        oof.loc[indices, "role"] = oof.loc[indices, "timestamp"].map(roles)

    selection_metrics = {}
    for name in candidate_reports:
        part = oof.loc[(oof["candidate"] == name) & (oof["target"] == "kpx_group_3") & (oof["role"] == "selection")]
        selection_metrics[name] = evaluate_group(part["y_true"], part["y_pred"], CAPACITY_KWH["kpx_group_3"]).to_dict()
    selected = max(selection_metrics, key=lambda name: selection_metrics[name]["score"])
    group3 = oof.loc[(oof["candidate"] == selected) & (oof["target"] == "kpx_group_3")].copy()
    group3["policy"] = selected
    scale, offset, strength, calibration_metrics = _affine_from_calibration_rows(group3, CAPACITY_KWH["kpx_group_3"])
    group3_affine = group3.copy()
    group3_affine["y_pred"] = np.clip(group3_affine["y_pred"] * (1 + strength * (scale - 1)) + strength * offset, 0.0, CAPACITY_KWH["kpx_group_3"])

    policies = {
        "pooled_group3_raw": _compose_with_safe(group3, safe, "pooled_group3_raw"),
        "pooled_group3_affine": _compose_with_safe(group3_affine, safe, "pooled_group3_affine"),
    }
    evaluations = {name: _evaluate_policy(policy) for name, policy in policies.items()}
    deltas = {name: _policy_delta(value, safe_evaluation) for name, value in evaluations.items()}
    bootstraps: dict[str, Any] = {}
    for i, (name, policy) in enumerate(policies.items(), start=1):
        base_g3 = safe.loc[(safe["target"] == "kpx_group_3") & (prediction_day(safe["timestamp"]).year == 2024)].sort_values("timestamp")
        cand_g3 = policy.loc[(policy["target"] == "kpx_group_3") & (prediction_day(policy["timestamp"]).year == 2024)].sort_values("timestamp")
        bootstraps[name] = {
            "group3": issue_day_block_bootstrap(base_g3["timestamp"], base_g3["y_true"], base_g3["y_pred"], cand_g3["y_pred"], CAPACITY_KWH["kpx_group_3"], repetitions=args.bootstrap_repetitions, seed=args.seed + i),
            "macro": macro_issue_day_bootstrap(safe, policy, args.bootstrap_repetitions, args.seed + 100 + i),
        }
    eligible = [name for name in policies if deltas[name]["groups"]["kpx_group_3"]["score"] > 0 and bootstraps[name]["group3"]["q05"] >= 0 and bootstraps[name]["macro"]["q05"] >= 0]
    best = max(eligible, key=lambda name: evaluations[name]["macro"]["score"]) if eligible else None
    report = {
        "experiment_id": "p4_pooled_group3_20260804", "candidate_reports": candidate_reports,
        "selection_2023_h1": selection_metrics, "selected_representation": selected,
        "calibration_2023_h2": {"scale": scale, "offset": offset, "strength": strength, "metrics": calibration_metrics},
        "safe_baseline_2024": safe_evaluation, "evaluations_2024": evaluations,
        "deltas_vs_safe": deltas, "bootstraps": bootstraps, "promotion_eligible": eligible,
        "selected_best": best, "runtime_seconds": time.perf_counter() - started,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(oof, artifact_dir / "oof_predictions.csv", {"selection": "2023 H1 group3", "calibration": "2023 H2 group3", "evaluation": "2024"})
    for name in policies:
        keep = name == best
        append_experiment_result(
            args.results_csv,
            {
                "experiment_id": f"p4_20260804__{name}", "hypothesis": "capacity-normalized pooled weather representation transfers group-1/2 knowledge to group 3",
                "feature_set": ["base", selected], "model": "pooled LightGBM normalized target",
                "hyperparameters": {"n_estimators": args.n_estimators, "selected_representation": selected, "affine": report["calibration_2023_h2"] if name.endswith("affine") else None},
                "fold_scores": candidate_reports[selected]["folds"], "group_metrics": evaluations[name]["groups"], "mean_delta": deltas[name]["macro"],
                "worst_fold_delta": deltas[name]["groups"]["kpx_group_3"]["score"], "bootstrap_interval": bootstraps[name],
                "runtime_seconds": report["runtime_seconds"], "decision": "keep" if keep else "reject",
                "reason": "passed locked group3 and macro bootstrap gates" if keep else "did not beat the safe group3 member with non-negative q05",
            },
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
