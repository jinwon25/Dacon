"""Promote only the stable group-3 portion of the P5 SCADA physical path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.postprocess_robust_cv import compose_by_group, macro_issue_day_bootstrap
from experiments.robust_cv_suite import _evaluate_policy, _policy_delta
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH
from src.validation import issue_day_block_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", default="artifacts/robust_cv_p2")
    parser.add_argument("--p5-dir", default="artifacts/scada_physical")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    p2_dir, p5_dir = Path(args.p2_dir), Path(args.p5_dir)
    report_path = p5_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    safe = pd.read_csv(p2_dir / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    safe = safe.loc[safe["policy"] == "safe_cv_best"].copy()
    full = pd.read_csv(p5_dir / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    gated = compose_by_group("scada_group3_gated", {
        "kpx_group_1": safe,
        "kpx_group_2": safe,
        "kpx_group_3": full,
    })
    baseline_eval = _evaluate_policy(safe)
    gated_eval = _evaluate_policy(gated)
    delta = _policy_delta(gated_eval, baseline_eval)
    group_bootstrap = {}
    for group_i, target in enumerate(CAPACITY_KWH, start=1):
        base = safe.loc[(safe["target"] == target) & (prediction_day(safe["timestamp"]).year == 2024)].sort_values("timestamp")
        candidate = gated.loc[(gated["target"] == target) & (prediction_day(gated["timestamp"]).year == 2024)].sort_values("timestamp")
        group_bootstrap[target] = issue_day_block_bootstrap(
            base["timestamp"], base["y_true"], base["y_pred"], candidate["y_pred"], CAPACITY_KWH[target],
            repetitions=args.bootstrap_repetitions, seed=args.seed + group_i,
        )
    macro = macro_issue_day_bootstrap(safe, gated, args.bootstrap_repetitions, args.seed + 100)
    promoted = group_bootstrap["kpx_group_3"]["q05"] > 0 and macro["q05"] > 0
    report["group3_gated"] = {
        "contract": "groups 1/2 byte-equivalent to safe_cv_best; only group 3 receives 27.5% SCADA physical blend",
        "evaluation_2024": gated_eval,
        "delta_vs_safe": delta,
        "bootstrap": {"groups": group_bootstrap, "macro": macro},
        "promotion_eligible": promoted,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(pd.concat([full, gated], ignore_index=True), p5_dir / "composite_oof_predictions.csv", {"policies": ["scada_physical_blend", "scada_group3_gated"]})
    append_experiment_result(args.results_csv, {
        "experiment_id": "p5_scada_group3_gated_20260804",
        "hypothesis": "SCADA physical path is complementary only for the group-3 bottleneck",
        "feature_set": ["safe_cv_best", "group3 training-only SCADA physical path"],
        "model": "group-gated OOF blend",
        "hyperparameters": {"kpx_group_3_physical_weight": 0.275, "kpx_group_1_2_weight": 0.0},
        "fold_scores": gated_eval,
        "group_metrics": gated_eval["groups"],
        "mean_delta": delta["macro"],
        "worst_fold_delta": min(value["score"] for value in delta["groups"].values()),
        "bootstrap_interval": report["group3_gated"]["bootstrap"],
        "runtime_seconds": 0.0,
        "decision": "keep" if promoted else "reject",
        "reason": "group3 and macro day-block q05 are positive; unchanged groups remain byte-equivalent" if promoted else "day-block stability gate failed",
    })
    print(json.dumps(report["group3_gated"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
