"""Expanding-origin confirmation for the issue-trajectory TCN.

The static year-forward TCN is intentionally conservative but mismatched to
production, where all 2024 labels are available before fitting 2025.  This
diagnostic restores that symmetry: every 2024 quarter is predicted by a fresh
model trained only on issues preceding that quarter.  The Q1 quantile/blend
choice remains frozen for Q2, Q3, and Q4.

This script is diagnostic only.  It never writes a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.issue_trajectory_tcn import (
    QUANTILES,
    TARGETS,
    SequenceBundle,
    fit_predict,
    flatten_prediction,
    load_issue_series,
    make_sequence_bundle,
    select_policy,
    select_trajectory_columns,
)
from experiments.kma_year_forward_quantile_blend import (
    apply_bounded_blend,
    metric_delta,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
FOLD_BOUNDS = (
    ("q1", "2024-01-01", "2024-04-01"),
    ("q2", "2024-04-01", "2024-07-01"),
    ("q3", "2024-07-01", "2024-10-01"),
    ("q4", "2024-10-01", "2025-01-01"),
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_bundle(
    bundle: SequenceBundle,
    rows: np.ndarray,
) -> SequenceBundle:
    return SequenceBundle(
        features=bundle.features[rows],
        targets=bundle.targets[rows] if bundle.targets is not None else None,
        target_mask=(
            bundle.target_mask[rows]
            if bundle.target_mask is not None
            else None
        ),
        timestamps=bundle.timestamps[rows],
        issues=bundle.issues[rows],
    )


def expanding_fold_rows(
    timestamps: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    first = timestamps[:, 0]
    folds: list[dict[str, Any]] = []
    for name, start_text, end_text in FOLD_BOUNDS:
        start = np.datetime64(start_text)
        end = np.datetime64(end_text)
        train = first < start
        query = (first >= start) & (first < end)
        if not train.any() or not query.any():
            raise ValueError(f"empty expanding fold: {name}")
        folds.append(
            {
                "name": name,
                "start": start_text,
                "end": end_text,
                "train": train,
                "query": query,
            }
        )
    return tuple(folds)


def run(args: argparse.Namespace) -> dict[str, Any]:
    features = pd.read_pickle(_rooted(args.features_train))
    columns = select_trajectory_columns(features)
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    issue = load_issue_series(_rooted(args.gfs_train))
    bundle = make_sequence_bundle(features[columns], issue, labels)
    folds = expanding_fold_rows(bundle.timestamps)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds:
        raise ValueError("at least one seed is required")

    raw_cache = _rooted(args.raw_cache)
    base_raw_cache = (
        _rooted(args.base_raw_cache) if args.base_raw_cache else None
    )
    base_retained = (
        np.load(base_raw_cache, allow_pickle=False)
        if base_raw_cache is not None
        else None
    )
    base_training = (
        json.loads(str(base_retained["training_json"].item()))
        if base_retained is not None
        else {}
    )
    training: dict[str, list[dict[str, Any]]] = {}
    fold_predictions: dict[str, np.ndarray] = {}
    fold_timestamps: dict[str, np.ndarray] = {}
    if raw_cache.exists():
        retained = np.load(raw_cache, allow_pickle=False)
        training = json.loads(str(retained["training_json"].item()))
        for fold in folds:
            name = fold["name"]
            fold_predictions[name] = retained[
                f"{name}__prediction"
            ].astype(float)
            fold_timestamps[name] = retained[f"{name}__timestamps"]
    else:
        payload: dict[str, np.ndarray] = {}
        for fold_position, fold in enumerate(folds):
            name = fold["name"]
            train = subset_bundle(bundle, fold["train"])
            query = subset_bundle(bundle, fold["query"])
            seed_predictions: list[np.ndarray] = []
            training[name] = list(base_training.get(name, []))
            if base_retained is not None:
                base_timestamps = base_retained[f"{name}__timestamps"]
                if not np.array_equal(base_timestamps, query.timestamps):
                    raise ValueError(
                        f"base raw cache timestamps differ for {name}"
                    )
                seed_predictions.append(
                    base_retained[f"{name}__prediction"].astype(float)
                )
            for seed in seeds:
                prediction, record = fit_predict(
                    train,
                    query,
                    seed=seed + 1_000 * fold_position,
                    hidden=args.hidden,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                seed_predictions.append(prediction)
                training[name].append(record)
            mean_prediction = np.mean(seed_predictions, axis=0)
            fold_predictions[name] = mean_prediction
            fold_timestamps[name] = query.timestamps
            payload[f"{name}__prediction"] = mean_prediction.astype(
                "float32"
            )
            payload[f"{name}__timestamps"] = query.timestamps
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        payload["training_json"] = np.asarray(
            json.dumps(training, allow_nan=False)
        )
        np.savez_compressed(raw_cache, **payload)

    baseline, truth_series, baseline_index, _ = (
        load_frozen_validation_baselines(
            _rooted(args.primary_cache),
            _rooted(args.residual_cache),
            _rooted(args.group3_cache),
        )
    )
    all_prediction = np.concatenate(
        [fold_predictions[fold["name"]] for fold in folds],
        axis=0,
    )
    all_timestamps = np.concatenate(
        [fold_timestamps[fold["name"]] for fold in folds],
        axis=0,
    )
    common = baseline_index.intersection(
        pd.DatetimeIndex(all_timestamps.reshape(-1))
    )
    period_rows = {
        name: np.asarray(
            (common >= pd.Timestamp(start))
            & (common < pd.Timestamp(end))
        )
        for name, start, end in FOLD_BOUNDS
    }
    period_rows["h2"] = period_rows["q3"] | period_rows["q4"]
    period_rows["full"] = np.ones(len(common), dtype=bool)

    validation: dict[str, Any] = {}
    for target_position, target in enumerate(TARGETS):
        capacity = CAPACITY_KWH[target]
        truth = truth_series[target].reindex(common).to_numpy(dtype=float)
        reference = baseline[target].reindex(common).to_numpy(dtype=float)
        experts = {
            quantile: flatten_prediction(
                all_prediction,
                all_timestamps,
                target_position,
                quantile_position,
                common,
            )
            * capacity
            for quantile_position, quantile in enumerate(QUANTILES)
        }
        spec, selection_records = select_policy(
            truth,
            reference,
            experts,
            capacity,
            period_rows["q1"],
        )
        candidate = apply_bounded_blend(
            reference,
            experts[spec["quantile"]],
            weight=spec["weight"],
            capacity=capacity,
        )
        deltas = {
            name: metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                rows,
            )
            for name, rows in period_rows.items()
        }
        monthly_h2 = {
            str(month): metric_delta(
                truth,
                reference,
                candidate,
                capacity,
                period_rows["h2"] & np.asarray(common.month == month),
            )
            for month in range(7, 13)
        }
        bootstrap = evaluate_blocked_rolling(
            truth,
            reference,
            candidate,
            common,
            pd.DatetimeIndex(issue.reindex(common)),
            period_rows["h2"] & (truth >= 0.10 * capacity),
            n_bootstrap=args.n_bootstrap,
            seed=20260727 + target_position,
        )
        positive_months = sum(
            row["score"] > 0.0 for row in monthly_h2.values()
        )
        gates = {
            "q1_selection_all_components_positive": bool(
                spec["selection_eligible"]
            ),
            "q2_all_components_positive": bool(
                min(deltas["q2"].values()) > 0.0
            ),
            "q3_all_components_positive": bool(
                min(deltas["q3"].values()) > 0.0
            ),
            "q4_all_components_positive": bool(
                min(deltas["q4"].values()) > 0.0
            ),
            "h2_all_components_positive": bool(
                min(deltas["h2"].values()) > 0.0
            ),
            "positive_h2_score_months_at_least_five": bool(
                positive_months >= 5
            ),
            "issue_bootstrap_q05_positive": bool(
                bootstrap["issue_block_bootstrap"]["q05"] > 0.0
            ),
            "issue_bootstrap_positive_fraction_at_least_95pct": bool(
                bootstrap["issue_block_bootstrap"]["positive_fraction"]
                >= 0.95
            ),
        }
        validation[target] = {
            "selected_spec": spec,
            "selection_records": selection_records,
            "period_deltas": deltas,
            "monthly_h2_deltas": monthly_h2,
            "positive_h2_score_months": int(positive_months),
            "issue_block_validation": bootstrap,
            "gates": gates,
            "promotion": "promoted" if all(gates.values()) else "rejected",
        }

    promoted = [
        target
        for target, result in validation.items()
        if result["promotion"] == "promoted"
    ]
    report = {
        "family": "issue_trajectory_multigroup_quantile_tcn_expanding",
        "method": (
            "fresh issue-level TCN fit before each 2024 quarter; Q1-only "
            "quantile/blend selection; frozen Q2/Q3/Q4 confirmation"
        ),
        "contract": {
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "submission_written": False,
            "folds": [
                {
                    "name": fold["name"],
                    "train_issues": int(fold["train"].sum()),
                    "query_issues": int(fold["query"].sum()),
                    "query_start": fold["start"],
                    "query_end": fold["end"],
                }
                for fold in folds
            ],
        },
        "architecture": {
            "features": int(len(columns)),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seeds": list(seeds),
            "base_raw_cache": (
                base_raw_cache.relative_to(ROOT).as_posix()
                if base_raw_cache is not None
                else None
            ),
        },
        "training": training,
        "validation": validation,
        "promoted_targets": promoted,
        "raw_cache": {
            "path": raw_cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(raw_cache),
        },
        "decision": (
            "promising screen; rerun with three fixed seeds before production"
            if promoted
            else "reject expanding trajectory TCN screen"
        ),
    }
    output = _rooted(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-train",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seeds", default="17")
    parser.add_argument(
        "--base-raw-cache",
        default="",
        help=(
            "Optional prior single-seed raw cache to include in the ensemble "
            "without retraining it."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--raw-cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_expanding_seed17_20260727.npz"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "issue_trajectory_tcn_expanding_seed17_20260727.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "promoted_targets": report["promoted_targets"],
                "selected_specs": {
                    target: result["selected_spec"]
                    for target, result in report["validation"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
