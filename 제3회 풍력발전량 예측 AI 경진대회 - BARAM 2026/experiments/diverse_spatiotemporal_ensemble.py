"""P6: audit a frozen 20% spatiotemporal member on the new robust base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.postprocess_robust_cv import compose_by_group, macro_issue_day_bootstrap
from experiments.robust_cv_suite import _evaluate_policy, _policy_delta
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_group
from src.validation import issue_day_block_bootstrap


def _neural_group3(seed_paths: list[Path]) -> pd.DataFrame:
    parts = []
    expected_index = None
    expected_truth = None
    for path in seed_paths:
        arrays = np.load(path, allow_pickle=True)
        timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["timestamps_ns"]))
        truth = arrays["truth"][:, 2].astype(float)
        # The cached neural target is capacity-normalized.  Convert it back to
        # the kWh scale before blending with the direct/SCADA predictions.
        prediction = arrays["prediction"][:, 2].astype(float) * CAPACITY_KWH["kpx_group_3"]
        if expected_index is None:
            expected_index, expected_truth = timestamps, truth
        elif not timestamps.equals(expected_index) or not np.allclose(truth, expected_truth, equal_nan=True):
            raise ValueError("Spatiotemporal seed OOF caches are not aligned")
        parts.append(prediction)
    return pd.DataFrame({"timestamp": expected_index, "neural_prediction": np.mean(parts, axis=0)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p5-dir", default="artifacts/scada_physical")
    parser.add_argument("--seed-oof", action="append", default=[])
    parser.add_argument("--artifact-dir", default="artifacts/diverse_ensemble")
    parser.add_argument("--results-csv", default="artifacts/experiment_results.csv")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    seed_paths = [Path(path) for path in args.seed_oof] or [
        Path("artifacts_final/spatiotemporal/validation_predictions.npz"),
        Path("artifacts_final/spatiotemporal_seed29/validation_predictions.npz"),
    ]
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    policies = pd.read_csv(Path(args.p5_dir) / "composite_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp"])
    robust = policies.loc[policies["policy"] == "scada_group3_gated"].copy()
    neural = _neural_group3(seed_paths)
    group3 = robust.loc[robust["target"] == "kpx_group_3"].merge(neural, on="timestamp", how="left", validate="one_to_one")
    if group3["neural_prediction"].isna().any():
        raise ValueError("Missing spatiotemporal OOF rows after alignment")
    group3["y_pred"] = np.clip(
        (1.0 - args.alpha) * group3["y_pred"] + args.alpha * group3["neural_prediction"],
        0.0,
        CAPACITY_KWH["kpx_group_3"],
    )
    candidate = compose_by_group("diverse_ensemble", {
        "kpx_group_1": robust,
        "kpx_group_2": robust,
        "kpx_group_3": group3,
    })
    baseline_eval = _evaluate_policy(robust)
    candidate_eval = _evaluate_policy(candidate)
    delta = _policy_delta(candidate_eval, baseline_eval)

    target = "kpx_group_3"
    capacity = CAPACITY_KWH[target]
    base_g3 = robust.loc[robust["target"] == target].sort_values("timestamp")
    candidate_g3 = candidate.loc[candidate["target"] == target].sort_values("timestamp")
    half_metrics = {}
    for name, mask in {
        "2024_h1_development_diagnostic": prediction_day(base_g3["timestamp"]).month <= 6,
        "2024_h2_locked_confirmation": prediction_day(base_g3["timestamp"]).month > 6,
    }.items():
        half_metrics[name] = {
            "baseline": evaluate_group(base_g3.loc[mask, "y_true"], base_g3.loc[mask, "y_pred"], capacity).to_dict(),
            "candidate": evaluate_group(candidate_g3.loc[mask, "y_true"], candidate_g3.loc[mask, "y_pred"], capacity).to_dict(),
        }
        half_metrics[name]["delta"] = {
            key: half_metrics[name]["candidate"][key] - half_metrics[name]["baseline"][key]
            for key in ("score", "one_minus_nmae", "ficr")
        }
    group_bootstrap = issue_day_block_bootstrap(
        base_g3["timestamp"], base_g3["y_true"], base_g3["y_pred"], candidate_g3["y_pred"], capacity,
        repetitions=args.bootstrap_repetitions, seed=args.seed,
    )
    macro_bootstrap = macro_issue_day_bootstrap(robust, candidate, args.bootstrap_repetitions, args.seed + 1)
    eligible = base_g3["y_true"].to_numpy() >= 0.10 * capacity
    base_error = base_g3["y_pred"].to_numpy() - base_g3["y_true"].to_numpy()
    neural_error = group3.sort_values("timestamp")["neural_prediction"].to_numpy() - base_g3["y_true"].to_numpy()
    half_score_deltas = [entry["delta"]["score"] for entry in half_metrics.values()]
    promotion = (
        delta["macro"]["score"] >= 0.001
        and sum(value > 0 for value in half_score_deltas) > len(half_score_deltas) / 2
        and min(half_score_deltas) >= -0.002
        and delta["groups"][target]["score"] > 0
        and macro_bootstrap["positive_fraction"] >= 0.80
        and macro_bootstrap["q05"] > 0
    )
    report = {
        "experiment_id": "p6_diverse_spatiotemporal_20260804",
        "selection_contract": {
            "alpha": args.alpha,
            "alpha_source": "frozen global_a20 selected in existing Q1/Q2 issue-blocked report; not retuned on this base",
            "changed_group": target,
            "seed_oof": [path.as_posix() for path in seed_paths],
        },
        "baseline_2024": baseline_eval,
        "candidate_2024": candidate_eval,
        "delta_vs_scada_group3_gated": delta,
        "half_metrics": half_metrics,
        "bootstrap": {"group3": group_bootstrap, "macro": macro_bootstrap},
        "complementarity": {
            "eligible_residual_correlation": float(np.corrcoef(base_error[eligible], neural_error[eligible])[0, 1]),
            "eligible_prediction_correlation": float(np.corrcoef(base_g3.loc[eligible, "y_pred"], group3.sort_values("timestamp").loc[eligible, "neural_prediction"])[0, 1]),
        },
        "promotion_eligible": promotion,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof_predictions(candidate, artifact_dir / "oof_predictions.csv", report["selection_contract"])
    append_experiment_result(args.results_csv, {
        "experiment_id": report["experiment_id"],
        "hypothesis": "a frozen two-seed spatial-temporal NWP model diversifies the group-3 direct/SCADA ensemble",
        "feature_set": ["robust base", "SCADA physical", "spatial-temporal NWP graph"],
        "model": "80% robust group3 + 20% two-seed spatial-temporal model",
        "hyperparameters": report["selection_contract"],
        "fold_scores": {"half_metrics": half_metrics, "full": candidate_eval},
        "group_metrics": candidate_eval["groups"],
        "mean_delta": delta["macro"],
        "worst_fold_delta": half_metrics["2024_h2_locked_confirmation"]["delta"]["score"],
        "bootstrap_interval": report["bootstrap"],
        "runtime_seconds": 0.0,
        "decision": "candidate_only" if promotion else "reject",
        "reason": (
            "frozen diversity weight improves both half-folds and has positive macro issue-day q05; "
            "kept as a higher-risk diverse candidate because locked-H2 FICR is slightly negative"
            if promotion else "score, worst-half, group-stability, or bootstrap gate failed"
        ),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
