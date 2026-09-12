"""Compose the controlled G3 expanding-trajectory TCN public probe.

The fixed q=0.65, weight=0.05 policy was found during a repeated 2024
robustness audit.  It is therefore controlled exploratory rather than a strict
OOF promotion.  The production fit uses every available 2022-2024 issue and
changes only group 3 of the frozen public incumbent, making one public score
an exact single-factor test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import evaluate_blocked_rolling
from experiments.issue_trajectory_tcn import (
    QUANTILES,
    SEEDS,
    TARGETS,
    fit_predict,
    flatten_prediction,
    load_issue_series,
    make_sequence_bundle,
    select_trajectory_columns,
)
from experiments.issue_trajectory_tcn_expanding import FOLD_BOUNDS
from experiments.kma_year_forward_quantile_blend import (
    apply_bounded_blend,
    metric_delta,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_3"
QUANTILE = 0.65
WEIGHT = 0.05
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_single_group_factor(
    incumbent: pd.DataFrame,
    expert: np.ndarray,
    *,
    target: str,
    weight: float,
) -> pd.DataFrame:
    if target not in CAPACITY_KWH:
        raise ValueError(f"unknown target: {target}")
    expert = np.asarray(expert, dtype=float)
    if expert.shape != (len(incumbent),):
        raise ValueError("expert and incumbent row counts differ")
    output = incumbent.copy()
    output[target] = apply_bounded_blend(
        incumbent[target].to_numpy(dtype=float),
        expert,
        weight=weight,
        capacity=CAPACITY_KWH[target],
    )
    for other in TARGETS:
        if other == target:
            continue
        if not np.array_equal(
            output[other].to_numpy(dtype=float),
            incumbent[other].to_numpy(dtype=float),
        ):
            raise AssertionError(f"non-target group changed: {other}")
    return output


def validation_evidence(args: argparse.Namespace) -> dict[str, Any]:
    retained = np.load(
        _rooted(args.expanding_validation_cache),
        allow_pickle=False,
    )
    prediction = np.concatenate(
        [
            retained[f"{name}__prediction"]
            for name, _, _ in FOLD_BOUNDS
        ],
        axis=0,
    )
    timestamps = np.concatenate(
        [
            retained[f"{name}__timestamps"]
            for name, _, _ in FOLD_BOUNDS
        ],
        axis=0,
    )
    baselines, truths, index, _ = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    common = index.intersection(
        pd.DatetimeIndex(timestamps.reshape(-1))
    )
    target_position = TARGETS.index(TARGET)
    quantile_position = QUANTILES.index(QUANTILE)
    capacity = CAPACITY_KWH[TARGET]
    expert = flatten_prediction(
        prediction,
        timestamps,
        target_position,
        quantile_position,
        common,
    ) * capacity
    truth = truths[TARGET].reindex(common).to_numpy(dtype=float)
    reference = baselines[TARGET].reindex(common).to_numpy(dtype=float)
    candidate = apply_bounded_blend(
        reference,
        expert,
        weight=WEIGHT,
        capacity=capacity,
    )
    rows = {
        name: np.asarray(
            (common >= pd.Timestamp(start))
            & (common < pd.Timestamp(end))
        )
        for name, start, end in FOLD_BOUNDS
    }
    rows["h2"] = rows["q3"] | rows["q4"]
    rows["full"] = np.ones(len(common), dtype=bool)
    period_deltas = {
        name: metric_delta(
            truth,
            reference,
            candidate,
            capacity,
            mask,
        )
        for name, mask in rows.items()
    }
    monthly = {
        str(month): metric_delta(
            truth,
            reference,
            candidate,
            capacity,
            np.asarray(common.month == month),
        )
        for month in range(1, 13)
    }
    issue = load_issue_series(_rooted(args.gfs_train))
    bootstrap = evaluate_blocked_rolling(
        truth,
        reference,
        candidate,
        common,
        pd.DatetimeIndex(issue.reindex(common)),
        rows["h2"] & (truth >= 0.10 * capacity),
        n_bootstrap=args.n_bootstrap,
        seed=20260727,
    )
    movement = np.abs(candidate - reference)
    return {
        "period_deltas": period_deltas,
        "monthly_deltas": monthly,
        "positive_score_months": int(
            sum(row["score"] > 0.0 for row in monthly.values())
        ),
        "issue_block_validation": bootstrap,
        "movement": {
            "changed_rows": int(np.sum(movement > 1e-8)),
            "mean_capacity_ratio": float(movement.mean() / capacity),
            "p95_capacity_ratio": float(
                np.quantile(movement, 0.95) / capacity
            ),
            "maximum_capacity_ratio": float(movement.max() / capacity),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence = validation_evidence(args)
    features_train = pd.read_pickle(_rooted(args.features_train))
    features_test = pd.read_pickle(_rooted(args.features_test))
    columns = select_trajectory_columns(features_train)
    labels = pd.read_csv(
        _rooted(args.labels),
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    issue_train = load_issue_series(_rooted(args.gfs_train))
    issue_test = load_issue_series(_rooted(args.gfs_test))
    train_bundle = make_sequence_bundle(
        features_train[columns],
        issue_train,
        labels,
    )
    test_bundle = make_sequence_bundle(
        features_test.reindex(columns=columns),
        issue_test,
        None,
    )

    raw_cache = _rooted(args.production_cache)
    if raw_cache.exists():
        retained = np.load(raw_cache, allow_pickle=False)
        prediction = retained["prediction"].astype(float)
        training = json.loads(str(retained["training_json"].item()))
    else:
        predictions: list[np.ndarray] = []
        training: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_prediction, record = fit_predict(
                train_bundle,
                test_bundle,
                seed=seed,
                hidden=args.hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            predictions.append(seed_prediction)
            training.append(record)
        prediction = np.mean(predictions, axis=0)
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_cache,
            prediction=prediction.astype("float32"),
            timestamps=test_bundle.timestamps,
            training_json=np.asarray(
                json.dumps(training, allow_nan=False)
            ),
        )

    incumbent = pd.read_csv(
        _rooted(args.incumbent),
        encoding="utf-8-sig",
        parse_dates=["forecast_kst_dtm"],
    )
    output_index = pd.DatetimeIndex(incumbent["forecast_kst_dtm"])
    expert_series = pd.Series(
        prediction[
            :,
            :,
            TARGETS.index(TARGET),
            QUANTILES.index(QUANTILE),
        ].reshape(-1),
        index=pd.DatetimeIndex(test_bundle.timestamps.reshape(-1)),
    )
    aligned_expert = expert_series.reindex(output_index)
    fallback_rows = aligned_expert.isna().to_numpy()
    expert = aligned_expert.fillna(
        pd.Series(
            incumbent[TARGET].to_numpy(dtype=float)
            / CAPACITY_KWH[TARGET],
            index=output_index,
        )
    ).to_numpy(dtype=float) * CAPACITY_KWH[TARGET]
    output = compose_single_group_factor(
        incumbent,
        expert,
        target=TARGET,
        weight=WEIGHT,
    )
    movement = np.abs(
        output[TARGET].to_numpy(dtype=float)
        - incumbent[TARGET].to_numpy(dtype=float)
    )
    output["forecast_kst_dtm"] = output[
        "forecast_kst_dtm"
    ].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(
            f"CandidateValidator rejected TCN probe: {audit.errors}"
        )

    local_group_delta = evidence["period_deltas"]["full"]["score"]
    report = {
        "family": "public_probe_issue_trajectory_tcn_g3",
        "promotion_tier": "controlled_exploratory",
        "selection_caveat": (
            "q=0.65 and weight=0.05 were chosen after repeated inspection "
            "of 2024 expanding-origin results; public score was not used"
        ),
        "contract": {
            "target": TARGET,
            "quantile": QUANTILE,
            "weight": WEIGHT,
            "changes_only_one_group": True,
            "public_score_used_for_selection": False,
            "test_actual_generation_used": False,
            "public_effect_exactly_identifiable_from_one_probe": True,
        },
        "validation": evidence,
        "projection": {
            "incumbent_public_score": float(args.incumbent_public_score),
            "local_group_score_delta": float(local_group_delta),
            "macro_delta_if_local_transfers": float(
                local_group_delta / 3.0
            ),
            "public_score_if_local_transfers": float(
                args.incumbent_public_score + local_group_delta / 3.0
            ),
            "warning": (
                "The transfer projection is not a leaderboard prediction; "
                "prior novel G3 factors have reversed publicly."
            ),
        },
        "production": {
            "features": int(len(columns)),
            "training_issues": int(len(train_bundle.features)),
            "test_issues": int(len(test_bundle.features)),
            "fallback_incumbent_rows": int(fallback_rows.sum()),
            "fallback_policy": (
                "keep incumbent where a complete finite 24-hour trajectory "
                "was unavailable; no cross-issue interpolation"
            ),
            "seeds": list(SEEDS),
            "training": training,
            "raw_cache": {
                "path": raw_cache.relative_to(ROOT).as_posix(),
                "sha256": _sha256(raw_cache),
            },
        },
        "movement": {
            "changed_rows": int(np.sum(movement > 1e-8)),
            "mean_capacity_ratio": float(
                movement.mean() / CAPACITY_KWH[TARGET]
            ),
            "p95_capacity_ratio": float(
                np.quantile(movement, 0.95) / CAPACITY_KWH[TARGET]
            ),
            "maximum_capacity_ratio": float(
                movement.max() / CAPACITY_KWH[TARGET]
            ),
        },
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
            "candidate_validator": audit.to_dict(),
        },
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
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
    parser.add_argument(
        "--features-test",
        default="artifacts_final/feature_cache/features_test.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument("--gfs-test", default="data/test/gfs_test.csv")
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
    parser.add_argument(
        "--expanding-validation-cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_expanding_ensemble3_20260727.npz"
        ),
    )
    parser.add_argument(
        "--incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--incumbent-public-score",
        type=float,
        default=0.6461250914,
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--production-cache",
        default=(
            "artifacts_final/lineage/"
            "issue_trajectory_tcn_production_ensemble3_20260727.npz"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "public_probe_issue_tcn_expanding_g3_q65w05_20260727.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "public_probe_issue_tcn_expanding_g3_q65w05_20260727.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "projection": report["projection"],
                "movement": report["movement"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
