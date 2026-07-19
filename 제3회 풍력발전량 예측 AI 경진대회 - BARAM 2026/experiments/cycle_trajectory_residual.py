"""Issue-cycle trajectory residual probe (no submission side effect).

Hypothesis: the within-issue-cycle shape of group-3 NWP trajectories (hub wind,
direction, pressure) contains a small, non-affine residual signal left by the
rolling fine-policy incumbent. We build cycle aggregates only from rows
belonging to the same NWP issue (forecast time minus lead hour), fit a bounded
residual expert, and audit it with contiguous Q1->Q2 and H1->H2 issue-cycle splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.exact_oof_meta_gate_sweep import (
    _prepare_validation,
    apply_meta_gate,
    fit_probabilities,
    settlement_benefit_labels,
)
from src.metrics import evaluate_group


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts_final" / "structural_20260718"
CAPACITY = 21000.0
TARGET = "kpx_group_3"
WEATHER_COLUMNS = [
    "ldaps__kpx_group_3__hub_ws117__idw",
    "gfs__kpx_group_3__hub_ws117__idw",
    "ldaps__kpx_group_3__ws10__idw",
    "gfs__kpx_group_3__ws10__idw",
    "ldaps__kpx_group_3__surface_0_sp__idw",
    "gfs__kpx_group_3__surface_0_sp__idw",
    "ldaps__kpx_group_3__hub_dir_sin__idw",
    "ldaps__kpx_group_3__hub_dir_cos__idw",
    "gfs__kpx_group_3__hub_dir_sin__idw",
    "gfs__kpx_group_3__hub_dir_cos__idw",
]


def cycle_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create issue-safe per-row trajectory-shape features.

    A cycle is identified by forecast_time - lead_hour.  Mean/std/min/max are
    computed over the 24 forecasts in that issue, all of which are known when
    that issue is released; no target or observation is used.
    """
    index = X.index
    cycle = index - pd.to_timedelta(X["lead_hour"].astype(float), unit="h")
    F = pd.DataFrame(index=index)
    for column in WEATHER_COLUMNS:
        values = (
            X[column].astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        grouped = values.groupby(cycle)
        mean = grouped.transform("mean")
        F[column + "__cycle_mean"] = mean
        F[column + "__cycle_std"] = grouped.transform("std").fillna(0.0)
        F[column + "__cycle_min"] = grouped.transform("min")
        F[column + "__cycle_max"] = grouped.transform("max")
        F[column + "__cycle_dev"] = values - mean
    F["lead_hour"] = X["lead_hour"].astype(float)
    F["hour"] = X["hour"].astype(float)
    F["month"] = X["month"].astype(float)
    F["cycle_id_ns"] = cycle.astype("int64") / 1e18
    return F.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def metric_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    before = evaluate_group(y, base, CAPACITY)
    after = evaluate_group(y, candidate, CAPACITY)
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Replace a JSON artifact as one completed file per runner invocation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def fit_predict(
    F: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    train: np.ndarray,
    query: pd.DataFrame | None = None,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=17,
        early_stopping=False,
    )
    model.fit(F.loc[train], (y - base)[train])
    return model.predict(F if query is None else query)


def reconstruct_rolling_fine_surfaces(
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    features: np.ndarray,
    action: np.ndarray,
    index: pd.DatetimeIndex,
    threshold: float = 0.545,
    extra_alpha: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rebuild the fixed fine-policy incumbent without using H2 labels.

    Q1 probabilities are used for Q2, while H1 probabilities are used for H2.
    The returned pair is (training surface, evaluation surface), allowing the
    residual expert to fit only against a genuinely earlier incumbent.
    """
    q1 = index < pd.Timestamp("2024-04-01")
    q2 = (index >= pd.Timestamp("2024-04-01")) & (index < pd.Timestamp("2024-07-01"))
    h1 = index < pd.Timestamp("2024-07-01")
    q1_benefit = settlement_benefit_labels(truth, current, member, q1)
    q1_probability, _ = fit_probabilities(features, q1_benefit, q1 & action)
    fine_from_q1, _ = apply_meta_gate(
        current, member, action, q1_probability, threshold=threshold, extra_alpha=extra_alpha
    )
    h1_benefit = settlement_benefit_labels(truth, current, member, h1)
    h1_probability, _ = fit_probabilities(features, h1_benefit, h1 & action)
    fine_from_h1, _ = apply_meta_gate(
        current, member, action, h1_probability, threshold=threshold, extra_alpha=extra_alpha
    )
    train_surface = current.copy()
    train_surface[q2] = fine_from_q1[q2]
    eval_surface = fine_from_h1.copy()
    eval_surface[q1] = current[q1]
    eval_surface[q2] = fine_from_q1[q2]
    return train_surface, eval_surface, {"threshold": threshold, "extra_alpha": extra_alpha}


def audit_period(
    F: pd.DataFrame,
    index: pd.DatetimeIndex,
    y: np.ndarray,
    train_base: np.ndarray,
    evaluate_base: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
    alpha: float = 0.02,
) -> tuple[dict[str, object], np.ndarray]:
    delta_hat = fit_predict(F, y, train_base, train)
    # Non-affine bounded policy: only positive residual actions, tiny blend.
    gate = delta_hat > 0.0
    candidate = np.clip(evaluate_base + alpha * delta_hat * gate, 0.0, CAPACITY)
    base_metric = evaluate_group(y[evaluate], evaluate_base[evaluate], CAPACITY).to_dict()
    candidate_metric = evaluate_group(y[evaluate], candidate[evaluate], CAPACITY).to_dict()
    return (
        {
            "base": base_metric,
            "candidate": candidate_metric,
            "delta": metric_delta(y[evaluate], evaluate_base[evaluate], candidate[evaluate]),
            "changed_ratio": float(gate[evaluate].mean()),
            "mean_absolute_movement_kwh": float(np.mean(np.abs(candidate[evaluate] - evaluate_base[evaluate]))),
        },
        delta_hat,
    )


def issue_bootstrap(
    index: pd.DatetimeIndex,
    lead: pd.Series,
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    evaluate: np.ndarray,
    n_bootstrap: int = 500,
) -> dict[str, object]:
    cycles = index - pd.to_timedelta(lead.astype(float), unit="h")
    cycles_eval = np.asarray(pd.unique(cycles[evaluate]))
    positions = {cycle: np.flatnonzero((cycles[evaluate].to_numpy() == cycle)) for cycle in cycles_eval}
    y_eval, b_eval, c_eval = y[evaluate], base[evaluate], candidate[evaluate]
    rng = np.random.default_rng(7318)
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(cycles_eval, size=len(cycles_eval), replace=True)
        rows = np.concatenate([positions[cycle] for cycle in sampled])
        values.append(metric_delta(y_eval[rows], b_eval[rows], c_eval[rows])["score"])
    values = np.asarray(values, dtype=float)
    return {
        "n_issue_cycles": int(len(cycles_eval)),
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float(np.mean(values > 0.0)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lineage = np.load(ROOT / "artifacts_final" / "lineage" / "exact_driver_oof.npz")
    meta_cache = np.load(ROOT / "artifacts_final" / "meta_gate" / "meta_gate_cache.npz")
    labels, valid_index, truth, group_1, group_2, exact_base, member, current, meta_features_array, action = _prepare_validation(
        ROOT / "data" / "train" / "train_labels.csv",
        ROOT / "artifacts_final" / "lineage" / "exact_driver_oof.npz",
    )
    valid_index = pd.DatetimeIndex(valid_index)
    exact_base = np.asarray(exact_base, dtype=float)
    truth = np.asarray(truth, dtype=float)
    current = np.asarray(current, dtype=float)
    member = np.asarray(member, dtype=float)
    if not np.array_equal(valid_index.astype("int64"), meta_cache["valid_index_ns"]):
        raise ValueError("meta-gate cache index is not aligned with exact OOF index")
    # Reconstruct the rolling fine policy once, frozen at threshold=.545 and
    # extra_alpha=.50.  Q1 probability trains Q2; H1 probability trains H2.
    q1 = valid_index < pd.Timestamp("2024-04-01")
    q2 = (valid_index >= pd.Timestamp("2024-04-01")) & (valid_index < pd.Timestamp("2024-07-01"))
    h1 = valid_index < pd.Timestamp("2024-07-01")
    h2 = valid_index >= pd.Timestamp("2024-07-01")
    rolling_train, rolling_eval, fine_policy = reconstruct_rolling_fine_surfaces(
        truth, current, member, meta_features_array, action, valid_index
    )
    base = rolling_eval
    X_all = pickle.load(open(ROOT / "artifacts_final" / "feature_cache" / "features_train.pkl", "rb"))
    X = X_all.loc[valid_index]
    F = cycle_features(X)
    # Q1 -> Q2 is selection; H1 -> H2 is locked confirmation.
    q2_report, _ = audit_period(F, valid_index, truth, rolling_train, rolling_eval, q1, q2)
    h2_report, h2_delta = audit_period(F, valid_index, truth, rolling_train, rolling_eval, h1, h2)
    h2_candidate = np.clip(rolling_eval + 0.02 * h2_delta * (h2_delta > 0.0), 0.0, CAPACITY)
    h2_report["issue_bootstrap"] = issue_bootstrap(valid_index, X["lead_hour"], truth, rolling_eval, h2_candidate, h2)
    monthly = {}
    for month in range(7, 13):
        mask = h2 & (valid_index.month == month)
        if not mask.any():
            continue
        monthly[str(month)] = metric_delta(truth[mask], rolling_eval[mask], h2_candidate[mask])
    h2_report["monthly_deltas"] = monthly
    # Run the frozen H1 expert on the full 2025 test horizon for an audit-only
    # artifact.  No CSV is written and this vector is not promoted automatically.
    X_test = pickle.load(open(ROOT / "artifacts_final" / "feature_cache" / "features_test.pkl", "rb"))
    F_test = cycle_features(X_test)
    test_index = X_test.index
    test_submission = pd.read_csv(
        ROOT / "submissions" / "blend_best_crossg3_traj_meta_finesweep.csv",
        encoding="utf-8-sig",
    )
    test_incumbent = test_submission[TARGET].to_numpy(dtype=float)
    test_reference = meta_cache["test_candidate"].astype(float)
    test_delta = fit_predict(F, truth, rolling_train, h1, query=F_test)
    test_candidate = np.clip(
        test_incumbent + 0.02 * test_delta * (test_delta > 0.0), 0.0, CAPACITY
    )
    promotion = (
        all(value >= 0.0 for value in q2_report["delta"].values())
        and all(value >= 0.0 for value in h2_report["delta"].values())
        and h2_report["issue_bootstrap"]["positive_fraction"] >= 0.90
        and h2_report["issue_bootstrap"]["q05"] >= 0.0
        and all(value["score"] >= 0.0 for value in monthly.values())
    )
    report = {
        "runner": "experiments/cycle_trajectory_residual.py",
        "method": "issue-cycle NWP trajectory-shape residual expert",
        "hypothesis": "within-issue 24-hour trajectory shape captures non-affine group-3 residual regimes",
        "feature_contract": {
            "issue_key": "forecast_kst_dtm - lead_hour",
            "columns": WEATHER_COLUMNS,
            "uses_target_or_observation": False,
            "uses_future_valid_weather": False,
            "feature_count": int(F.shape[1]),
        },
        "selection_contract": {
            "development": "2024-Q1 train -> 2024-Q2 evaluate",
            "locked": "2024-H1 train -> 2024-H2 evaluate once",
            "policy": {"alpha": 0.02, "gate": "positive predicted residual only"},
        },
        "baseline_parity": {
            "source": "rolling reconstruction from exact_oof_meta_gate_sweep.py",
            "lineage_script": "experiments/exact_oof_meta_gate_sweep.py",
            "fine_policy": fine_policy,
            "exact_driver_surface": "artifacts_final/lineage/exact_driver_oof.npz::kpx_group_3__exact_base",
            "reference_cache_surface": "artifacts_final/meta_gate/meta_gate_cache.npz::valid_candidate (reference p=.55/a=.25; not fine)",
            "reference_rows_changed_vs_exact": int(np.sum(np.abs(meta_cache["valid_candidate"] - exact_base) > 1e-6)),
            "reference_changed_ratio_vs_exact": float(np.mean(np.abs(meta_cache["valid_candidate"] - exact_base) > 1e-6)),
            "reference_mae_kwh_vs_exact": float(np.mean(np.abs(meta_cache["valid_candidate"] - exact_base))),
            "rolling_fine_rows_changed_vs_exact": int(np.sum(np.abs(base - exact_base) > 1e-6)),
            "rolling_fine_changed_ratio_vs_exact": float(np.mean(np.abs(base - exact_base) > 1e-6)),
            "rolling_fine_mae_kwh_vs_exact": float(np.mean(np.abs(base - exact_base))),
            "test_finesweep_rows_changed_vs_reference": int(np.sum(np.abs(test_incumbent - test_reference) > 0.01)),
            "test_finesweep_changed_ratio_vs_reference": float(np.mean(np.abs(test_incumbent - test_reference) > 0.01)),
            "test_finesweep_mae_kwh_vs_reference": float(np.mean(np.abs(test_incumbent - test_reference))),
            "parity_note": "All locked deltas below are relative to rolling fine p=.545/a=.50; reference cache p=.55/a=.25 is retained only for parity audit. No affine probe is used.",
        },
        "q1_to_q2": q2_report,
        "h1_to_h2_locked": h2_report,
        "test_inference": {
            "index_start": str(test_index.min()),
            "index_end": str(test_index.max()),
            "rows": int(len(test_index)),
            "changed_ratio": float(np.mean(np.abs(test_candidate - test_incumbent) > 1e-6)),
            "mean_absolute_movement_kwh": float(np.mean(np.abs(test_candidate - test_incumbent))),
            "incumbent_mean_kwh": float(np.mean(test_incumbent)),
            "candidate_mean_kwh": float(np.mean(test_candidate)),
            "artifact": "test_predictions.npz",
        },
        "promotion": {
            "eligible": bool(promotion),
            "candidate_created": False,
            "reason": "issue-cycle bootstrap q05/positive-fraction and/or fold gates not met"
            if not promotion
            else "eligible; human review required before candidate generation",
        },
    }
    np.savez_compressed(
        OUT / "locked_predictions.npz",
        index_ns=valid_index.astype("int64").to_numpy(),
        truth=truth.astype(np.float32),
        base=base.astype(np.float32),
        locked_candidate=h2_candidate.astype(np.float32),
        locked_delta=h2_delta.astype(np.float32),
    )
    np.savez_compressed(
        OUT / "test_predictions.npz",
        index_ns=test_index.astype("int64").to_numpy(),
        incumbent=test_incumbent.astype(np.float32),
        candidate=test_candidate.astype(np.float32),
        delta=test_delta.astype(np.float32),
    )
    atomic_write_json(OUT / "structural_report.json", report)

    monthly_scores = [float(value["score"]) for value in monthly.values()]
    locked_movement_ratio = np.abs(h2_candidate[h2] - rolling_eval[h2]) / CAPACITY
    evaluation = {
        "family": "cycle_trajectory_residual",
        "family_group": "structural_residual",
        "direction": "positive_residual",
        "locked_score_delta": float(h2_report["delta"]["score"]),
        "locked_one_minus_nmae_delta": float(h2_report["delta"]["one_minus_nmae"]),
        "locked_ficr_delta": float(h2_report["delta"]["ficr"]),
        "expected_macro_score_delta": float(h2_report["delta"]["score"] / 3.0),
        "positive_months": int(sum(value >= 0.0 for value in monthly_scores)),
        "total_months": int(len(monthly_scores)),
        "bootstrap_positive_fraction": float(
            h2_report["issue_bootstrap"]["positive_fraction"]
        ),
        "bootstrap_q05": float(h2_report["issue_bootstrap"]["q05"]),
        "changed_ratio": float(h2_report["changed_ratio"]),
        "p95_movement_ratio": float(np.quantile(locked_movement_ratio, 0.95)),
        "worst_month_score_delta": float(min(monthly_scores)) if monthly_scores else None,
        "fold_scores": monthly_scores,
        "cv_mean": float(np.mean(monthly_scores)) if monthly_scores else None,
        "cv_std": float(np.std(monthly_scores)) if monthly_scores else None,
        "oof_path": "artifacts_final/structural_20260718/locked_predictions.npz",
        "test_path": "artifacts_final/structural_20260718/test_predictions.npz",
        "runtime_seconds": None,
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": float(h2_report["delta"]["score"] / 3.0),
        "selection_direction": "maximize",
        "notes": (
            "Issue-cycle trajectory features are forecast-time safe. Fine incumbent is "
            "rolling threshold=.545/extra_alpha=.50; reference cache p=.55/a=.25 is "
            "parity-only. Promotion rejected because at least one monthly fold is "
            "negative; no submission candidate produced."
            if not promotion
            else "Issue-cycle trajectory expert passed locked gates; candidate generation remains human-gated."
        ),
    }
    atomic_write_json(OUT / "agent_evaluation.json", evaluation)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
