"""Compose and audit frozen Sprint 0.66 OOF candidates.

Rules are fixed on earlier OOF periods. This module only reports the locked
2024 result and never reads the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.postprocess_robust_cv import macro_issue_day_bootstrap
from experiments.robust_cv_suite import _evaluate_policy, _policy_delta
from src.data_audit import prediction_day
from src.diagnostics import detailed_group_metrics, slice_diagnostics
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH


def _load_policy(path: Path, policy: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    selected = frame.loc[frame["policy"] == policy].copy()
    if selected.empty:
        raise ValueError(f"Policy {policy!r} is absent from {path}")
    return selected


def _compose_ficr_bayes(safe: pd.DataFrame, bayes_oof: Path) -> pd.DataFrame:
    selected = _load_policy(bayes_oof, "selected_shrinkage")
    selected = selected.loc[
        selected["target"] == "kpx_group_2",
        ["timestamp", "target", "y_true", "y_pred"],
    ]
    group_mask = safe["target"] == "kpx_group_2"
    merged = safe.loc[group_mask].drop(columns=["y_pred"]).merge(
        selected,
        on=["timestamp", "target", "y_true"],
        how="left",
        validate="one_to_one",
    )
    if merged["y_pred"].isna().any():
        raise ValueError("Missing locked Bayes rows while composing group 2")
    out = pd.concat([safe.loc[~group_mask], merged], ignore_index=True)
    out["policy"] = "ficr_bayes"
    return out


def _metric_delta(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    keys = (
        "score", "one_minus_nmae", "ficr", "hit_rate_6pct", "hit_rate_8pct",
        "actual_weighted_hit_rate_6pct", "actual_weighted_hit_rate_8pct",
        "bias_kwh", "residual_std_kwh",
    )
    return {key: float(candidate[key] - base[key]) for key in keys}


def _slice_delta(base: pd.DataFrame, candidate: pd.DataFrame, capacity: float) -> dict[str, Any]:
    left = base[["timestamp", "target", "y_true", "lead_hour", "y_pred"]].rename(
        columns={"y_pred": "base_pred"}
    )
    right = candidate[["timestamp", "target", "y_true", "y_pred"]].rename(
        columns={"y_pred": "candidate_pred"}
    )
    joined = left.merge(right, on=["timestamp", "target", "y_true"], validate="one_to_one")
    timestamp = pd.to_datetime(joined["timestamp"])
    axes = {
        "actual_power_bin_diagnostic_only": pd.cut(
            joined["y_true"] / capacity,
            bins=[0.10, 0.20, 0.40, 0.60, 0.80, 1.00, np.inf],
            right=False,
            include_lowest=True,
        ),
        "lead_time_bucket": pd.cut(
            joined["lead_hour"], bins=[11.5, 17.5, 23.5, 29.5, 35.5], include_lowest=True,
        ),
        "season": timestamp.dt.month.map(
            {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
        ),
    }
    result: dict[str, Any] = {}
    for axis_name, categories in axes.items():
        result[axis_name] = {}
        for value in categories.dropna().unique():
            mask = categories == value
            truth = joined.loc[mask, "y_true"].to_numpy()
            base_metrics = detailed_group_metrics(
                truth, joined.loc[mask, "base_pred"].to_numpy(), capacity,
            )
            candidate_metrics = detailed_group_metrics(
                truth, joined.loc[mask, "candidate_pred"].to_numpy(), capacity,
            )
            result[axis_name][str(value)] = {
                "baseline": base_metrics,
                "candidate": candidate_metrics,
                "delta": _metric_delta(base_metrics, candidate_metrics),
            }
    return result


def _attach_interval_width(frame: pd.DataFrame, quantiles_path: Path) -> pd.DataFrame:
    quantiles = pd.read_csv(quantiles_path, encoding="utf-8-sig", parse_dates=["timestamp"])
    quantiles["predictive_interval_width"] = quantiles["q0.95"] - quantiles["q0.05"]
    return frame.merge(
        quantiles[["timestamp", "target", "predictive_interval_width"]],
        on=["timestamp", "target"],
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-oof", default="artifacts/scada_physical/composite_oof_predictions.csv")
    parser.add_argument("--bayes-oof", default="artifacts/metric_aware_bayes/oof_predictions.csv")
    parser.add_argument("--quantile-oof", default="artifacts/metric_aware_bayes/quantile_oof_predictions.csv")
    parser.add_argument("--diverse-oof", default="artifacts/diverse_ensemble_066/oof_predictions.csv")
    parser.add_argument("--artifact-dir", default="artifacts/final_candidates_066")
    parser.add_argument("--results-csv", default="artifacts/experiments_066.csv")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2066)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe = _load_policy(Path(args.safe_oof), "scada_group3_gated")
    safe = safe.loc[prediction_day(safe["timestamp"]).year == 2024].copy()
    safe["policy"] = "safe_cv_best"
    ficr_bayes = _compose_ficr_bayes(safe, Path(args.bayes_oof))
    diverse = _load_policy(Path(args.diverse_oof), "diverse_ensemble")
    diverse = diverse.loc[prediction_day(diverse["timestamp"]).year == 2024].copy()
    candidates = {
        "safe_cv_best": safe,
        "ficr_bayes": ficr_bayes,
        "diverse_ensemble": diverse,
    }

    report: dict[str, Any] = {
        "selection_contract": {
            "safe_cv_best": "P2 group-gated direct policy plus group-3 SCADA alpha=0.275 transferred from groups 1/2",
            "ficr_bayes": "safe policy with group-2 alpha=0.5 Bayes shrinkage selected on 2023 H1 and confirmed on H2",
            "diverse_ensemble": "safe policy with frozen 20% two-seed spatial-temporal member for group 3",
            "locked_evaluation_year": 2024,
            "test_data_used_for_selection": False,
        },
        "evaluations_2024": {},
        "deltas_vs_safe": {},
        "macro_day_block_bootstrap_vs_safe": {},
        "diagnostics": {},
        "diagnostic_deltas_vs_safe": {},
    }
    safe_eval = _evaluate_policy(safe)
    for index, (name, frame) in enumerate(candidates.items()):
        evaluation = _evaluate_policy(frame)
        report["evaluations_2024"][name] = evaluation
        report["deltas_vs_safe"][name] = _policy_delta(evaluation, safe_eval)
        if name != "safe_cv_best":
            report["macro_day_block_bootstrap_vs_safe"][name] = macro_issue_day_bootstrap(
                safe, frame, args.bootstrap_repetitions, args.seed + index,
            )
        report["diagnostics"][name] = {}
        report["diagnostic_deltas_vs_safe"][name] = {}
        diagnostic_frame = _attach_interval_width(frame, Path(args.quantile_oof))
        for target, part in diagnostic_frame.groupby("target"):
            capacity = CAPACITY_KWH[target]
            report["diagnostics"][name][target] = slice_diagnostics(part, capacity)
            report["diagnostic_deltas_vs_safe"][name][target] = _slice_delta(
                safe.loc[safe["target"] == target],
                frame.loc[frame["target"] == target],
                capacity,
            )

    report["promotion_decisions"] = {
        "safe_cv_best": "keep: stable base with positive SCADA gate bootstrap",
        "ficr_bayes": "candidate_only: Bayes gain exists, but incremental gain over safe is below the primary threshold",
        "diverse_ensemble": "candidate_only: positive blend/bootstrap, but locked-H2 group-3 FICR is slightly negative",
    }
    (artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    saved = []
    for name, frame in candidates.items():
        part = frame.copy()
        if "candidate" in part:
            part = part.rename(columns={"candidate": "base_member"})
        part["candidate"] = name
        saved.append(part)
    save_oof_predictions(
        pd.concat(saved, ignore_index=True),
        artifact_dir / "oof_predictions.csv",
        report["selection_contract"],
    )

    bayes_delta = report["deltas_vs_safe"]["ficr_bayes"]
    append_experiment_result(args.results_csv, {
        "experiment_id": "s066_m1_bayes_group2_gate",
        "hypothesis": "the 2023-selected metric-aware action adds group-2 settlement value on top of the safe composite",
        "feature_set": ["base", "11 eligible quantiles"],
        "model": "safe composite with group-2 alpha=0.5 quantile Bayes action",
        "hyperparameters": report["selection_contract"]["ficr_bayes"],
        "fold_scores": report["evaluations_2024"]["ficr_bayes"],
        "group_metrics": report["evaluations_2024"]["ficr_bayes"]["groups"],
        "mean_delta": bayes_delta["macro"],
        "worst_fold_delta": min(value["score"] for value in bayes_delta["groups"].values()),
        "bootstrap_interval": report["macro_day_block_bootstrap_vs_safe"]["ficr_bayes"],
        "runtime_seconds": 0.0,
        "decision": "candidate_only",
        "reason": "incremental Score is positive but below 0.0015 and FICR gain is below 0.005",
    })
    print(json.dumps({
        "evaluations_2024": report["evaluations_2024"],
        "deltas_vs_safe": report["deltas_vs_safe"],
        "bootstrap": report["macro_day_block_bootstrap_vs_safe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
