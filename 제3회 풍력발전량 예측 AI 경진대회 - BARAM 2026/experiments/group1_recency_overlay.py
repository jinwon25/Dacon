"""Audit a bounded group-1 recency correction against the public incumbent.

The experiment isolates one production-relevant hypothesis: the change from
an equal-history direct model to a recent-12-month direct model transfers to
the next year.  The direct-model delta is added to the frozen incumbent OOF,
so this is not another broad replacement of the production pipeline.

Overlay strength is selected on 2024 H1 and evaluated once on locked H2.
Neither test weather nor leaderboard results are used for model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.compose_residual_stack_candidate import apply_capped_residual_stack
from experiments.incumbent_residual_noncrossing import interval_month
from src.data_audit import prediction_day
from src.experiment_tracking import append_experiment_result, save_oof_predictions
from src.metrics import CAPACITY_KWH, evaluate_group
from src.validation import issue_day_block_bootstrap
from train import make_model, select_feature_columns


TARGET = "kpx_group_1"
CAPACITY = CAPACITY_KWH[TARGET]
CURRENT_G1_RESIDUAL_WEIGHT = 0.1375
DEFAULT_ALPHAS = (0.0, 0.10, 0.25, 0.50, 1.0)


def compose_recency_overlay(
    incumbent: np.ndarray,
    equal_history: np.ndarray,
    recent_history: np.ndarray,
    *,
    alpha: float,
    capacity: float = CAPACITY,
) -> np.ndarray:
    """Add only the recent-minus-equal-history factor to an incumbent."""

    incumbent = np.asarray(incumbent, dtype=float)
    equal_history = np.asarray(equal_history, dtype=float)
    recent_history = np.asarray(recent_history, dtype=float)
    if not (incumbent.shape == equal_history.shape == recent_history.shape):
        raise ValueError("recency overlay inputs must have identical shapes")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not all(np.isfinite(values).all() for values in (incumbent, equal_history, recent_history)):
        raise ValueError("recency overlay inputs must be finite")
    return np.clip(incumbent + alpha * (recent_history - equal_history), 0.0, capacity)


def select_alpha(
    truth: np.ndarray,
    incumbent: np.ndarray,
    equal_history: np.ndarray,
    recent_history: np.ndarray,
    selection_rows: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    """Select by official Score, breaking near-ties toward less movement."""

    selection_rows = np.asarray(selection_rows, dtype=bool)
    if not selection_rows.any():
        raise ValueError("selection_rows is empty")
    if 0.0 not in alphas or any(alpha < 0.0 for alpha in alphas):
        raise ValueError("alphas must be non-negative and include zero")
    metrics: dict[str, dict[str, float | int]] = {}
    for alpha in sorted(set(alphas)):
        candidate = compose_recency_overlay(
            incumbent, equal_history, recent_history, alpha=alpha,
        )
        metrics[str(alpha)] = evaluate_group(
            truth[selection_rows], candidate[selection_rows], CAPACITY,
        ).to_dict()
    # The rounded comparison avoids choosing a larger movement for numerical
    # noise at the discontinuous metric cliffs.
    selected = max(
        sorted(set(alphas)),
        key=lambda alpha: (
            round(float(metrics[str(alpha)]["score"]), 12),
            round(float(metrics[str(alpha)]["ficr"]), 12),
            -alpha,
        ),
    )
    return float(selected), metrics


def _load_incumbent_oof(
    primary_path: Path,
    residual_path: Path,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        residual_path, allow_pickle=False,
    ) as residual:
        prefix = f"{TARGET}__"
        index = pd.DatetimeIndex(pd.to_datetime(primary[f"{prefix}index_ns"]))
        issue = pd.DatetimeIndex(pd.to_datetime(primary[f"{prefix}issue_ns"]))
        truth = primary[f"{prefix}truth"].astype(float)
        residual_index = pd.DatetimeIndex(pd.to_datetime(residual[f"{prefix}index_ns"]))
        residual_truth = residual[f"{prefix}truth"].astype(float)
        if not index.equals(residual_index):
            raise ValueError("primary and residual OOF indexes differ")
        if not np.allclose(truth, residual_truth, atol=2e-3, rtol=0.0):
            raise ValueError("primary and residual OOF truths differ")
        incumbent = apply_capped_residual_stack(
            primary[f"{prefix}reference"],
            primary[f"{prefix}candidate"],
            residual[f"{prefix}candidate"],
            residual_weight=CURRENT_G1_RESIDUAL_WEIGHT,
            capacity=CAPACITY,
            movement_cap_ratio=0.05,
        )
    return index, issue, truth, np.asarray(incumbent, dtype=float)


def _fit_direct(
    X: pd.DataFrame,
    y: pd.Series,
    train_rows: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
) -> Any:
    eligible = (
        np.asarray(train_rows, dtype=bool)
        & y.notna().to_numpy()
        & (y.to_numpy(dtype=float) >= 0.10 * CAPACITY)
    )
    if eligible.sum() < 1_000:
        raise ValueError(f"too few eligible training rows: {eligible.sum()}")
    model = make_model(seed=seed, n_estimators=n_estimators)
    model.fit(X.loc[eligible], y.loc[eligible], callbacks=[lgb.log_evaluation(0)])
    return model


def _metric_delta(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    base = evaluate_group(truth[rows], baseline[rows], CAPACITY)
    trial = evaluate_group(truth[rows], candidate[rows], CAPACITY)
    return {
        "score": float(trial.score - base.score),
        "one_minus_nmae": float(trial.one_minus_nmae - base.one_minus_nmae),
        "ficr": float(trial.ficr - base.ficr),
    }


def _movement(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(np.asarray(candidate) - np.asarray(baseline)) / CAPACITY
    return {
        "changed_fraction": float(np.mean(delta > 1e-12)),
        "mean_capacity_ratio": float(np.mean(delta)),
        "p95_capacity_ratio": float(np.quantile(delta, 0.95)),
        "max_capacity_ratio": float(np.max(delta)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X_all = pd.read_pickle(args.feature_cache)
    columns = select_feature_columns(X_all, TARGET, "base")
    X = X_all[columns]
    labels = pd.read_csv(
        data_dir / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").reindex(X.index)
    y = labels[TARGET]
    years = np.asarray(prediction_day(X.index).year)

    equal_model = _fit_direct(
        X, y, np.isin(years, (2022, 2023)),
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    recent_model = _fit_direct(
        X, y, years == 2023,
        seed=args.seed + 1,
        n_estimators=args.n_estimators,
    )

    index, issue, truth, incumbent = _load_incumbent_oof(
        Path(args.primary_cache), Path(args.residual_cache),
    )
    pred_frame = X.reindex(index)
    if pred_frame.isna().any(axis=None):
        missing = int(pred_frame.isna().any(axis=1).sum())
        raise ValueError(f"OOF feature alignment produced {missing} incomplete rows")
    equal_pred = np.clip(equal_model.predict(pred_frame), 0.0, CAPACITY)
    recent_pred = np.clip(recent_model.predict(pred_frame), 0.0, CAPACITY)

    # The first cached timestamp ends a 2023 interval. It is ineligible in the
    # supplied truth, but it is still excluded explicitly from selection and
    # locked evaluation to preserve the forecast-day contract.
    oof_year = np.asarray(prediction_day(index).year)
    month = interval_month(index)
    selection_rows = (oof_year == 2024) & (month <= 6)
    locked_rows = (oof_year == 2024) & (month >= 7)
    full_rows = oof_year == 2024
    alphas = tuple(float(value) for value in args.alphas.split(","))
    selected_alpha, selection_metrics = select_alpha(
        truth, incumbent, equal_pred, recent_pred, selection_rows, alphas,
    )
    candidate = compose_recency_overlay(
        incumbent, equal_pred, recent_pred, alpha=selected_alpha,
    )

    periods = {
        "2024_h1_selection": selection_rows,
        "2024_h2_locked": locked_rows,
        "2024_full": full_rows,
    }
    deltas = {
        name: _metric_delta(truth, incumbent, candidate, rows)
        for name, rows in periods.items()
    }
    direct_metrics = {
        "equal_history": evaluate_group(truth[full_rows], equal_pred[full_rows], CAPACITY).to_dict(),
        "recent_12_months": evaluate_group(truth[full_rows], recent_pred[full_rows], CAPACITY).to_dict(),
    }
    direct_delta = _metric_delta(truth, equal_pred, recent_pred, full_rows)
    bootstrap = issue_day_block_bootstrap(
        index[locked_rows],
        truth[locked_rows],
        incumbent[locked_rows],
        candidate[locked_rows],
        CAPACITY,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 2,
    )
    monthly = {
        str(month_number): _metric_delta(
            truth, incumbent, candidate,
            full_rows & (month == month_number),
        )
        for month_number in range(1, 13)
    }

    h2 = deltas["2024_h2_locked"]
    full = deltas["2024_full"]
    macro_full_score = full["score"] / 3.0
    stable = (
        selected_alpha > 0.0
        and h2["score"] > 0.0
        and h2["ficr"] >= 0.0
        and bootstrap["q05"] >= 0.0
    )
    promoted = stable and macro_full_score >= 0.0010
    decision = "keep" if promoted else ("candidate_only" if stable else "reject")
    if selected_alpha == 0.0:
        reason = "H1 selected zero recency movement"
    elif not stable:
        reason = "the H1-selected recency factor failed locked-H2 component/bootstrap stability"
    elif not promoted:
        reason = "locked transfer was stable but macro-equivalent full Score gain was below 0.001"
    else:
        reason = "positive H1/H2 transfer, non-negative locked FICR/q05, and macro-equivalent gain >= 0.001"

    report: dict[str, Any] = {
        "experiment_id": "s066_recency_g1_overlay_20260805",
        "hypothesis": "a recent-12-month minus equal-history direct-model factor improves the frozen group-1 incumbent",
        "selection_contract": {
            "selection": "2024 H1",
            "locked_confirmation": "2024 H2",
            "target": TARGET,
            "alphas": list(alphas),
            "selected_alpha": selected_alpha,
            "test_data_used": False,
            "leaderboard_used_for_parameter_selection": False,
        },
        "direct_model_comparison_2024": direct_metrics,
        "recent_minus_equal_direct_delta_2024": direct_delta,
        "selection_metrics": selection_metrics,
        "deltas_vs_current_incumbent": deltas,
        "monthly_deltas_vs_current_incumbent": monthly,
        "locked_h2_issue_day_bootstrap": bootstrap,
        "movement": _movement(incumbent[full_rows], candidate[full_rows]),
        "macro_equivalent_full_score_delta": macro_full_score,
        "decision": decision,
        "reason": reason,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    frames = []
    for name, values in (
        ("current_incumbent", incumbent),
        ("equal_history_direct", equal_pred),
        ("recent_12m_direct", recent_pred),
        ("recency_overlay", candidate),
    ):
        frames.append(pd.DataFrame({
            "timestamp": index[full_rows],
            "target": TARGET,
            "fold": "2023_to_2024",
            "candidate": name,
            "y_true": truth[full_rows],
            "y_pred": np.asarray(values)[full_rows],
            "issue": issue[full_rows],
        }))
    save_oof_predictions(
        pd.concat(frames, ignore_index=True),
        artifact_dir / "oof_predictions.csv",
        report["selection_contract"],
    )

    append_experiment_result(args.results_csv, {
        "experiment_id": report["experiment_id"],
        "hypothesis": report["hypothesis"],
        "feature_set": ["base", "recency factor only"],
        "model": "eligible-only LightGBM L1 recent12-minus-all overlay on frozen incumbent",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "alphas": list(alphas),
            "selected_alpha": selected_alpha,
        },
        "fold_scores": deltas,
        "group_metrics": direct_metrics,
        "mean_delta": {
            "score": macro_full_score,
            "one_minus_nmae": full["one_minus_nmae"] / 3.0,
            "ficr": full["ficr"] / 3.0,
        },
        "worst_fold_delta": min(deltas[name]["score"] / 3.0 for name in deltas),
        "bootstrap_interval": bootstrap,
        "runtime_seconds": report["runtime_seconds"],
        "decision": decision,
        "reason": reason,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--feature-cache",
        default="artifacts/robust_cv_p2/feature_cache/features_train.pkl",
    )
    parser.add_argument(
        "--primary-cache",
        default="artifacts_final/lineage/kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz",
    )
    parser.add_argument(
        "--residual-cache",
        default="artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz",
    )
    parser.add_argument("--artifact-dir", default="artifacts/group1_recency_overlay")
    parser.add_argument("--results-csv", default="artifacts/experiments_066.csv")
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--alphas", default="0,0.10,0.25,0.50,1.0")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20660805)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "selected_alpha": report["selection_contract"]["selected_alpha"],
        "direct_delta": report["recent_minus_equal_direct_delta_2024"],
        "overlay_deltas": report["deltas_vs_current_incumbent"],
        "bootstrap": report["locked_h2_issue_day_bootstrap"],
        "decision": report["decision"],
        "reason": report["reason"],
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
