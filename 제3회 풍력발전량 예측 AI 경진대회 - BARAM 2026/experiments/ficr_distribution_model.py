from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import select_feature_columns


QUANTILES = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90])


def make_quantile_model(alpha: float, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
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


def ficr_decision(quantile_predictions: np.ndarray, capacity: float, mean_generation: float) -> np.ndarray:
    """Choose the point maximizing expected competition utility.

    The five quantiles are treated as equal-mass representatives of the
    conditional distribution among evaluation-eligible observations.
    """
    samples = np.sort(np.asarray(quantile_predictions, dtype=float), axis=1)
    shifts = np.asarray([0.0, -0.08, -0.06, 0.06, 0.08]) * capacity
    candidates = np.concatenate([samples + shift for shift in shifts], axis=1)
    candidates = np.clip(candidates, 0.0, capacity)

    errors = np.abs(candidates[:, :, None] - samples[:, None, :]) / capacity
    unit_fraction = np.where(errors <= 0.06, 1.0, np.where(errors <= 0.08, 0.75, 0.0))
    generation_weight = samples[:, None, :] / max(float(mean_generation), 1.0)
    utility = (-0.5 * errors + 0.5 * generation_weight * unit_fraction).mean(axis=2)
    return candidates[np.arange(len(candidates)), np.argmax(utility, axis=1)]


def select_postprocess(
    y_true: np.ndarray,
    median_pred: np.ndarray,
    decision_pred: np.ndarray,
    capacity: float,
    selection_mask: np.ndarray,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        raw = median_pred + alpha * (decision_pred - median_pred)
        for scale in (0.98, 0.99, 1.00, 1.01, 1.02):
            for offset in (-300.0, -150.0, 0.0, 150.0, 300.0):
                pred = np.clip(raw * scale + offset, 0.0, capacity)
                metric = evaluate_group(y_true[selection_mask], pred[selection_mask], capacity)
                if best is None or metric.score > best["selection_score"]:
                    best = {
                        "alpha": alpha,
                        "scale": scale,
                        "offset": offset,
                        "selection_score": metric.score,
                    }
    assert best is not None
    return best


def apply_postprocess(
    median_pred: np.ndarray,
    decision_pred: np.ndarray,
    capacity: float,
    selected: dict[str, float],
) -> np.ndarray:
    raw = median_pred + selected["alpha"] * (decision_pred - median_pred)
    return np.clip(raw * selected["scale"] + selected["offset"], 0.0, capacity)


def monthly_comparison(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    capacity: float,
) -> dict[str, object]:
    rows = []
    for month in range(1, 13):
        mask = timestamps.month == month
        base_metric = evaluate_group(y_true[mask], baseline[mask], capacity)
        candidate_metric = evaluate_group(y_true[mask], candidate[mask], capacity)
        rows.append(
            {
                "month": month,
                "baseline_score": base_metric.score,
                "candidate_score": candidate_metric.score,
                "delta": candidate_metric.score - base_metric.score,
            }
        )
    return {"months_improved": sum(row["delta"] > 0 for row in rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="artifacts_feature_cache")
    parser.add_argument("--artifact-dir", default="artifacts_ficr_distribution")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--selection-end", default="2024-07-01 00:00:00")
    parser.add_argument("--n-estimators", type=int, default=700)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    X_all = load_or_build_features(args.data_dir, "train", args.cache_dir)
    labels = pd.read_csv(Path(args.data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_all.index)

    valid_start = pd.Timestamp(args.valid_start)
    selection_end = pd.Timestamp(args.selection_end)
    report: dict[str, object] = {
        "quantiles": QUANTILES.tolist(),
        "valid_start": args.valid_start,
        "selection_end": args.selection_end,
        "n_estimators": args.n_estimators,
        "targets": {},
    }
    baseline_predictions: dict[str, np.ndarray] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    truth: dict[str, np.ndarray] = {}
    cache: dict[str, np.ndarray] = {}

    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        columns = select_feature_columns(X_all, target, "base")
        X = X_all[columns]
        train_mask = (X.index < valid_start) & y.notna() & (y >= 0.10 * capacity)
        valid_mask = (X.index >= valid_start) & y.notna()
        valid_index = X.index[valid_mask]
        valid_quantiles = []

        for quantile_i, quantile in enumerate(QUANTILES, start=1):
            model = make_quantile_model(
                alpha=float(quantile),
                seed=31_000 + 100 * target_i + quantile_i,
                n_estimators=args.n_estimators,
            )
            model.fit(X.loc[train_mask], y.loc[train_mask], callbacks=[lgb.log_evaluation(0)])
            valid_quantiles.append(model.predict(X.loc[valid_mask]))

        quantile_matrix = np.sort(np.column_stack(valid_quantiles), axis=1)
        median_pred = np.clip(quantile_matrix[:, 2], 0.0, capacity)
        decision_pred = ficr_decision(
            quantile_matrix,
            capacity=capacity,
            mean_generation=float(y.loc[train_mask].mean()),
        )
        y_valid = y.loc[valid_mask].to_numpy(dtype=float)
        first_half = valid_index < selection_end
        second_half = ~first_half

        baseline_post = select_postprocess(
            y_valid, median_pred, median_pred, capacity, first_half
        )
        candidate_post = select_postprocess(
            y_valid, median_pred, decision_pred, capacity, first_half
        )
        baseline_pred = apply_postprocess(median_pred, median_pred, capacity, baseline_post)
        candidate_pred = apply_postprocess(median_pred, decision_pred, capacity, candidate_post)

        baseline_h2 = evaluate_group(y_valid[second_half], baseline_pred[second_half], capacity)
        candidate_h2 = evaluate_group(y_valid[second_half], candidate_pred[second_half], capacity)
        baseline_full = evaluate_group(y_valid, baseline_pred, capacity)
        candidate_full = evaluate_group(y_valid, candidate_pred, capacity)
        monthly = monthly_comparison(
            valid_index, y_valid, baseline_pred, candidate_pred, capacity
        )
        report["targets"][target] = {
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "baseline_postprocess": baseline_post,
            "candidate_postprocess": candidate_post,
            "baseline_h2": baseline_h2.to_dict(),
            "candidate_h2": candidate_h2.to_dict(),
            "h2_score_delta": candidate_h2.score - baseline_h2.score,
            "baseline_full": baseline_full.to_dict(),
            "candidate_full": candidate_full.to_dict(),
            "full_score_delta": candidate_full.score - baseline_full.score,
            "monthly": monthly,
        }
        baseline_predictions[target] = baseline_pred[second_half]
        candidate_predictions[target] = candidate_pred[second_half]
        truth[target] = y_valid[second_half]
        cache[f"{target}__valid_index_ns"] = valid_index.astype("int64").to_numpy()
        cache[f"{target}__valid_truth"] = y_valid.astype("float32")
        cache[f"{target}__valid_quantiles"] = quantile_matrix.astype("float32")
        cache[f"{target}__baseline_pred"] = baseline_pred.astype("float32")
        cache[f"{target}__candidate_pred"] = candidate_pred.astype("float32")
        print(target, json.dumps(report["targets"][target], ensure_ascii=False), flush=True)

    report["h2_baseline_competition"] = evaluate_competition(truth, baseline_predictions)
    report["h2_candidate_competition"] = evaluate_competition(truth, candidate_predictions)
    report["h2_competition_score_delta"] = (
        report["h2_candidate_competition"]["score"]
        - report["h2_baseline_competition"]["score"]
    )
    (artifact_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(artifact_dir / "validation_cache.npz", **cache)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("h2_")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
