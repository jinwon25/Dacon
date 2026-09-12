from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.postprocess_robust_cv import macro_issue_day_bootstrap
from experiments.robust_cv_suite import (
    CANDIDATES,
    _apply_affine,
    _evaluate_policy,
    _fit_fixed_lgbm,
    _policy_delta,
    _representative_weather,
)
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.features import build_features
from src.metrics import CAPACITY_KWH, evaluate_group
from src.validation import assign_oof_roles, issue_day_block_bootstrap, make_expanding_year_folds
from train import select_feature_columns


BLOCKS = ("site_aware", "trajectory", "physical")


def _load_block_features(data_dir: Path, cache_dir: Path, block: str) -> pd.DataFrame:
    path = cache_dir / f"features_{block}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_blocks = ("base",) if block == "site_aware" else ("base", block)
    frame = build_features(data_dir, "train", feature_blocks=feature_blocks)
    frame.to_pickle(path)
    return frame


def _columns(X: pd.DataFrame, target: str, block: str) -> list[str]:
    base = select_feature_columns(X, target, "base")
    if block == "site_aware":
        return select_feature_columns(X, target, "own_idw_nohub")
    prefix = f"{block}__"
    extras = []
    for col in X.columns:
        if not col.startswith(prefix):
            continue
        if "__kpx_group_" not in col or f"__{target}__" in col:
            extras.append(col)
    return list(dict.fromkeys(base + extras))


def _candidate_for_target(target: str):
    name = "e2_eligible_weight100" if target == "kpx_group_3" else "e1_eligible_uniform"
    return next(candidate for candidate in CANDIDATES if candidate.name == name)


def _compose_policy(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    raw = raw.copy()
    raw["policy"] = name + "__raw"
    affine, _ = _apply_affine(raw)
    parts = []
    for target in CAPACITY_KWH:
        source = affine if target == "kpx_group_1" else raw
        part = source.loc[source["target"] == target].copy()
        part["policy"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _per_group_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    from src.data_audit import prediction_day

    output = {}
    for target in CAPACITY_KWH:
        base = baseline.loc[(baseline["target"] == target) & (prediction_day(baseline["timestamp"]).year == 2024)].sort_values("timestamp")
        contender = candidate.loc[(candidate["target"] == target) & (prediction_day(candidate["timestamp"]).year == 2024)].sort_values("timestamp")
        output[target] = issue_day_block_bootstrap(
            base["timestamp"], base["y_true"], base["y_pred"], contender["y_pred"], CAPACITY_KWH[target],
            repetitions=repetitions, seed=seed + int(target[-1]),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="One-block-at-a-time P3 ablation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--p2-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--artifact-dir", default="artifacts/feature_ablation")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")
    p2_oof = pd.read_csv(Path(args.p2_dir) / "oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    composite = pd.read_csv(Path(args.p2_dir) / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    baseline_policy = composite.loc[composite["policy"] == "safe_cv_best"].copy()
    baseline_evaluation = _evaluate_policy(baseline_policy)

    # The source-specific structure block is exactly redundant in this release:
    # both sources use the same 13:00 cycle, lead and one-hour cutoff margin.
    structure = build_features(data_dir, "train", feature_blocks=("base", "forecast_structure"))
    structure_cols = [col for col in structure if col.endswith(("__lead_hour", "__forecast_cycle_hour", "__cutoff_margin_hour"))]
    structure_summary = {col: {"nunique": int(structure[col].nunique()), "matches_generic_lead": bool(col.endswith("__lead_hour") and np.array_equal(structure[col].to_numpy(), structure["lead_hour"].to_numpy()))} for col in structure_cols}
    append_experiment_result(
        args.results_csv,
        {
            "experiment_id": "p3_forecast_structure_20260804",
            "hypothesis": "source-specific lead and forecast-cycle metadata add information beyond actual generic lead",
            "feature_set": ["base", "forecast_structure"],
            "model": "not fitted; deterministic redundancy audit",
            "hyperparameters": structure_summary,
            "fold_scores": {}, "group_metrics": {}, "mean_delta": {"score": 0.0, "one_minus_nmae": 0.0, "ficr": 0.0},
            "worst_fold_delta": 0.0, "bootstrap_interval": {}, "runtime_seconds": 0.0,
            "decision": "reject", "reason": "source leads duplicate generic lead and cycle/margin are constant",
        },
    )

    report: dict[str, Any] = {"forecast_structure": {"decision": "reject", "columns": structure_summary}, "blocks": {}}
    all_oof = []
    for block_i, block in enumerate(BLOCKS, start=1):
        print(f"BUILD {block}", flush=True)
        X_all = _load_block_features(data_dir, artifact_dir / "feature_cache", block)
        weather = _representative_weather(X_all)
        block_rows = []
        fold_metrics: dict[str, Any] = {}
        feature_counts: dict[str, int] = {}
        for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
            y = labels[target].reindex(X_all.index)
            columns = _columns(X_all, target, block)
            feature_counts[target] = len(columns)
            X = X_all[columns]
            candidate = _candidate_for_target(target)
            fold_metrics[target] = {}
            for fold_i, fold in enumerate(make_expanding_year_folds(X.index, y, target), start=1):
                # Match the corresponding P2 candidate seed so the feature block
                # is the only changed hypothesis.
                candidate_i = 6 if target == "kpx_group_3" else 2
                seed = args.seed + target_i * 1_000 + fold_i * 100 + candidate_i
                print(f"FIT {block} {target} {fold.name} features={len(columns)}", flush=True)
                model = _fit_fixed_lgbm(X, y, fold.train, candidate, capacity, seed, args.n_estimators)
                pred = np.clip(model.predict(X.loc[fold.valid]), 0.0, capacity)
                truth = y.loc[fold.valid].to_numpy(dtype=float)
                fold_metrics[target][fold.name] = evaluate_group(truth, pred, capacity).to_dict()
                part = pd.DataFrame({
                    "timestamp": X.index[fold.valid], "target": target, "fold": fold.name,
                    "candidate": block, "y_true": truth, "y_pred": pred,
                })
                for col in weather:
                    part[col] = weather.loc[fold.valid, col].to_numpy()
                block_rows.append(part)
        raw = pd.concat(block_rows, ignore_index=True)
        for target, indices in raw.groupby("target").groups.items():
            unique_index = pd.DatetimeIndex(sorted(raw.loc[indices, "timestamp"].unique()))
            role_by_time = dict(zip(unique_index, assign_oof_roles(unique_index)))
            raw.loc[indices, "role"] = raw.loc[indices, "timestamp"].map(role_by_time)
        policy = _compose_policy(raw, f"p3_{block}")
        evaluation = _evaluate_policy(policy)
        delta = _policy_delta(evaluation, baseline_evaluation)
        selection_delta: dict[str, float] = {}
        for target in ("kpx_group_1", "kpx_group_2"):
            base_name = "e1_eligible_uniform"
            base = p2_oof.loc[(p2_oof["target"] == target) & (p2_oof["candidate"] == base_name) & (p2_oof["role"] == "selection")]
            contender = raw.loc[(raw["target"] == target) & (raw["role"] == "selection")]
            selection_delta[target] = evaluate_group(contender["y_true"], contender["y_pred"], CAPACITY_KWH[target]).score - evaluate_group(base["y_true"], base["y_pred"], CAPACITY_KWH[target]).score
        group_bootstrap = _per_group_bootstrap(baseline_policy, policy, args.bootstrap_repetitions, args.seed + block_i * 100)
        macro_bootstrap = macro_issue_day_bootstrap(baseline_policy, policy, args.bootstrap_repetitions, args.seed + block_i)
        keep = (
            all(value >= 0 for value in selection_delta.values())
            and delta["macro"]["score"] > 0
            and min(value["score"] for value in delta["groups"].values()) >= 0
            and macro_bootstrap["q05"] >= 0
        )
        block_report = {
            "feature_counts": feature_counts, "fold_metrics": fold_metrics,
            "selection_score_delta": selection_delta, "evaluation_2024": evaluation,
            "delta_vs_safe_cv_best": delta, "group_bootstrap": group_bootstrap,
            "macro_bootstrap": macro_bootstrap, "decision": "keep" if keep else "reject",
        }
        report["blocks"][block] = block_report
        all_oof.append(policy)
        append_experiment_result(
            args.results_csv,
            {
                "experiment_id": f"p3_{block}_20260804", "hypothesis": f"the {block} feature block adds stable forward signal",
                "feature_set": ["base", block], "model": "LightGBM L1 with locked P2 target policy",
                "hyperparameters": {"n_estimators": args.n_estimators, "feature_counts": feature_counts},
                "fold_scores": fold_metrics, "group_metrics": evaluation["groups"], "mean_delta": delta["macro"],
                "worst_fold_delta": min(value["score"] for value in delta["groups"].values()),
                "bootstrap_interval": {"groups": group_bootstrap, "macro": macro_bootstrap},
                "runtime_seconds": time.perf_counter() - started, "decision": "keep" if keep else "reject",
                "reason": "passed selection, every-group, and macro-bootstrap gates" if keep else "failed selection transfer, group stability, or macro-bootstrap gate",
            },
        )
        print(json.dumps({block: block_report}, ensure_ascii=False, indent=2), flush=True)
    report["runtime_seconds"] = time.perf_counter() - started
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(pd.concat(all_oof, ignore_index=True), artifact_dir / "oof_predictions.csv", {"blocks": list(BLOCKS), "baseline": "robust_cv_p2/safe_cv_best"})


if __name__ == "__main__":
    main()
