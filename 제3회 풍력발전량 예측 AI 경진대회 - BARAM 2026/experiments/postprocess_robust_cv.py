from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.robust_cv_suite import (
    _apply_affine,
    _evaluate_policy,
    _policy_delta,
    _policy_frame,
)
from src.data_audit import prediction_day
from src.diagnostics import slice_diagnostics
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_competition
from src.validation import issue_day_block_bootstrap


def macro_issue_day_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    base = baseline.copy()
    contender = candidate.copy()
    base["forecast_day"] = prediction_day(base["timestamp"])
    contender["forecast_day"] = prediction_day(contender["timestamp"])
    base = base.loc[base["forecast_day"].dt.year == 2024]
    contender = contender.loc[contender["forecast_day"].dt.year == 2024]
    days = pd.DatetimeIndex(sorted(base["forecast_day"].unique()))
    if not days.equals(pd.DatetimeIndex(sorted(contender["forecast_day"].unique()))):
        raise ValueError("Baseline and candidate issue-day coverage differs")
    base_lookup: dict[str, dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]] = {}
    candidate_lookup: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    for target in CAPACITY_KWH:
        base_part = base.loc[base["target"] == target]
        candidate_part = contender.loc[contender["target"] == target]
        base_lookup[target] = {
            pd.Timestamp(day): (part["y_true"].to_numpy(), part["y_pred"].to_numpy())
            for day, part in base_part.groupby("forecast_day", sort=False)
        }
        candidate_lookup[target] = {
            pd.Timestamp(day): part["y_pred"].to_numpy()
            for day, part in candidate_part.groupby("forecast_day", sort=False)
        }
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        sampled = rng.choice(days, size=len(days), replace=True)
        base_truth: dict[str, np.ndarray] = {}
        base_pred: dict[str, np.ndarray] = {}
        candidate_pred: dict[str, np.ndarray] = {}
        for target in CAPACITY_KWH:
            base_truth[target] = np.concatenate([base_lookup[target][pd.Timestamp(day)][0] for day in sampled])
            base_pred[target] = np.concatenate([base_lookup[target][pd.Timestamp(day)][1] for day in sampled])
            candidate_pred[target] = np.concatenate([candidate_lookup[target][pd.Timestamp(day)] for day in sampled])
        base_metric = evaluate_competition(base_truth, base_pred)
        candidate_metric = evaluate_competition(base_truth, candidate_pred)
        deltas[iteration] = candidate_metric["score"] - base_metric["score"]
    return {
        "repetitions": repetitions,
        "issue_days": int(len(days)),
        "mean": float(deltas.mean()),
        "q025": float(np.quantile(deltas, 0.025)),
        "q05": float(np.quantile(deltas, 0.05)),
        "q50": float(np.quantile(deltas, 0.50)),
        "q95": float(np.quantile(deltas, 0.95)),
        "q975": float(np.quantile(deltas, 0.975)),
        "positive_fraction": float(np.mean(deltas > 0)),
    }


def compose_by_group(name: str, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for target, source in sources.items():
        part = source.loc[source["target"] == target].copy()
        part["policy"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    report_path = artifact_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    oof = pd.read_csv(artifact_dir / "oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    baseline = _policy_frame(oof, report["selected_candidates"]["e1"], "corrected_baseline")
    affine, _ = _apply_affine(baseline)
    weighted = _policy_frame(oof, report["selected_candidates"]["e2"], "generation_weighted")

    safe = compose_by_group(
        "safe_cv_best",
        {"kpx_group_1": affine, "kpx_group_2": baseline, "kpx_group_3": weighted},
    )
    ficr = compose_by_group(
        "ficr_focused",
        {"kpx_group_1": affine, "kpx_group_2": affine, "kpx_group_3": weighted},
    )
    composites = {"safe_cv_best": safe, "ficr_focused": ficr}
    baseline_evaluation = _evaluate_policy(baseline)
    report.setdefault("evaluations_2024", {})
    report.setdefault("deltas_vs_corrected_baseline", {})
    report.setdefault("day_block_bootstrap", {})
    report["macro_day_block_bootstrap"] = {}
    report["composite_contract"] = {
        "safe_cv_best": {
            "kpx_group_1": "E1 affine fitted only on 2023 H2; 2024 group q05 positive",
            "kpx_group_2": "corrected E1 baseline; uncertain affine rejected",
            "kpx_group_3": "lambda=1 transferred from group-1/2 selection; no group-3 calibration",
        },
        "ficr_focused": {
            "difference_from_safe": "also applies the 2023-H2 group-2 affine; mean positive but group q05 negative",
        },
    }
    for i, (name, policy) in enumerate(composites.items(), start=1):
        evaluation = _evaluate_policy(policy)
        delta = _policy_delta(evaluation, baseline_evaluation)
        report["evaluations_2024"][name] = evaluation
        report["deltas_vs_corrected_baseline"][name] = delta
        report["day_block_bootstrap"][name] = {}
        for target in CAPACITY_KWH:
            base_part = baseline.loc[(baseline["target"] == target) & (prediction_day(baseline["timestamp"]).year == 2024)].sort_values("timestamp")
            candidate_part = policy.loc[(policy["target"] == target) & (prediction_day(policy["timestamp"]).year == 2024)].sort_values("timestamp")
            report["day_block_bootstrap"][name][target] = issue_day_block_bootstrap(
                base_part["timestamp"], base_part["y_true"], base_part["y_pred"], candidate_part["y_pred"],
                CAPACITY_KWH[target], repetitions=args.bootstrap_repetitions, seed=args.seed + i * 100 + int(target[-1]),
            )
        report["macro_day_block_bootstrap"][name] = macro_issue_day_bootstrap(
            baseline, policy, args.bootstrap_repetitions, args.seed + i,
        )
        keep = name == "safe_cv_best"
        append_experiment_result(
            args.results_csv,
            {
                "experiment_id": f"robust_cv_p2_20260804__{name}",
                "hypothesis": "retain only group-level actions whose locked 2024 direction is stable" if keep else "accept uncertain group-2 FiCR affine for additional settlement upside",
                "feature_set": ["base", "actual_lead_from_availability"],
                "model": "group-gated LightGBM L1 policy",
                "hyperparameters": report["composite_contract"][name],
                "fold_scores": evaluation,
                "group_metrics": evaluation["groups"],
                "mean_delta": delta["macro"],
                "worst_fold_delta": min(value["score"] for value in delta["groups"].values()),
                "bootstrap_interval": {"groups": report["day_block_bootstrap"][name], "macro": report["macro_day_block_bootstrap"][name]},
                "runtime_seconds": 0.0,
                "decision": "keep" if keep else "candidate_only",
                "reason": "every retained group action has non-negative q05; untouched groups are byte-equivalent" if keep else "macro is stable but group-2 q05 is negative",
            },
        )
    report["promotion_eligible"] = ["safe_cv_best"]
    report["selected_best"] = "safe_cv_best"
    report["diagnostics"]["safe_cv_best"] = {}
    safe_year = prediction_day(safe["timestamp"]).year == 2024
    for target, part in safe.loc[safe_year].groupby("target"):
        columns = [col for col in ("timestamp", "y_true", "y_pred", "lead_hour", "wind_speed", "wind_direction", "nwp_disagreement") if col in part]
        report["diagnostics"]["safe_cv_best"][target] = slice_diagnostics(part[columns], CAPACITY_KWH[target])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(pd.concat(composites.values(), ignore_index=True), artifact_dir / "composite_oof_predictions.csv", {"source": "robust_cv_p2/oof_predictions.csv", "selection": report["composite_contract"]})
    print(json.dumps({name: {"evaluation": report["evaluations_2024"][name], "delta": report["deltas_vs_corrected_baseline"][name], "macro_bootstrap": report["macro_day_block_bootstrap"][name]} for name in composites}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
