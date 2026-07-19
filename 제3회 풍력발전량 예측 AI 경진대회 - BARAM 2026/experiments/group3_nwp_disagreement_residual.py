from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group
from src.physical_signals import build_group_physical_signals


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
SELECTION_START = pd.Timestamp("2024-05-01 00:00:00")
LOCKED_START = pd.Timestamp("2024-07-01 01:00:00")


@dataclass(frozen=True)
class CorrectionPolicy:
    alpha: float
    max_correction_ratio: float
    max_nwp_spread: float
    min_direction_agreement: float


def _model(seed: int, n_estimators: int = 600) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=45,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def _learning_frame(signals: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    frame = signals.reindex(base.index).copy()
    frame["base_ratio"] = base.to_numpy(dtype=float) / CAPACITY
    frame["base_ratio_sq"] = frame["base_ratio"] ** 2
    frame["lead_hour"] = (((base.index.hour - 1) % 24) + 12).astype(float)
    frame["hour_sin"] = np.sin(2.0 * np.pi * base.index.hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * base.index.hour / 24.0)
    frame["doy_sin"] = np.sin(2.0 * np.pi * base.index.dayofyear / 365.25)
    frame["doy_cos"] = np.cos(2.0 * np.pi * base.index.dayofyear / 365.25)
    return frame.replace([np.inf, -np.inf], np.nan)


def apply_policy(
    base: np.ndarray,
    correction_ratio: np.ndarray,
    signal_frame: pd.DataFrame,
    policy: CorrectionPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    correction = np.clip(
        correction_ratio,
        -policy.max_correction_ratio,
        policy.max_correction_ratio,
    )
    gate = (
        (signal_frame["nwp_hub_ws_rel_spread"].to_numpy() <= policy.max_nwp_spread)
        & (
            signal_frame["nwp_direction_agreement"].to_numpy()
            >= policy.min_direction_agreement
        )
        & (base >= 0.08 * CAPACITY)
    )
    candidate = np.asarray(base, dtype=float).copy()
    candidate[gate] += policy.alpha * CAPACITY * correction[gate]
    return np.clip(candidate, 0.0, CAPACITY), gate


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _compare(truth: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    before = evaluate_group(truth, base, CAPACITY)
    after = evaluate_group(truth, candidate, CAPACITY)
    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "delta": _metric_delta(before, after),
    }


def select_policy(
    truth: np.ndarray,
    base: np.ndarray,
    correction_ratio: np.ndarray,
    signals: pd.DataFrame,
) -> tuple[CorrectionPolicy, list[dict[str, object]]]:
    baseline = evaluate_group(truth, base, CAPACITY)
    candidates: list[dict[str, object]] = []
    for alpha in (0.05, 0.10, 0.20, 0.30):
        for max_correction in (0.01, 0.02, 0.04):
            # 10.0 is effectively unbounded for this normalized feature while
            # keeping the JSON report standards-compliant (no Infinity token).
            for max_spread in (0.10, 0.20, 0.40, 10.0):
                for min_direction in (-1.0, 0.0, 0.5):
                    policy = CorrectionPolicy(
                        alpha, max_correction, max_spread, min_direction
                    )
                    prediction, gate = apply_policy(
                        base, correction_ratio, signals, policy
                    )
                    metric = evaluate_group(truth, prediction, CAPACITY)
                    delta = _metric_delta(baseline, metric)
                    candidates.append(
                        {
                            "policy": asdict(policy),
                            "changed_rows": int(gate.sum()),
                            "changed_ratio": float(gate.mean()),
                            "metric": metric.to_dict(),
                            "delta": delta,
                            "passes_component_signs": (
                                delta["one_minus_nmae"] >= 0.0
                                and delta["ficr"] >= 0.0
                            ),
                        }
                    )
    candidates.sort(
        key=lambda item: (
            item["passes_component_signs"],
            item["delta"]["score"],
            item["delta"]["ficr"],
            -item["changed_ratio"],
        ),
        reverse=True,
    )
    return CorrectionPolicy(**candidates[0]["policy"]), candidates


def _bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    n_bootstrap: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(20260718)
    normalized_days = index.normalize()
    days = normalized_days.unique()
    positions = {day: np.flatnonzero(normalized_days == day) for day in days}
    deltas = []
    for _ in range(n_bootstrap):
        sample = rng.choice(days, len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sample])
        deltas.append(
            evaluate_group(truth[rows], candidate[rows], CAPACITY).score
            - evaluate_group(truth[rows], base[rows], CAPACITY).score
        )
    values = np.asarray(deltas)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((values > 0.0).mean()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def run_experiment(
    data_dir: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
    artifact_dir: Path,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not index.equals(driver_index):
        raise ValueError("Meta and exact-driver validation indexes differ")
    base = pd.Series(meta["valid_candidate"].astype(float), index=index)
    truth = pd.Series(driver[f"{TARGET}__valid_truth"].astype(float), index=index)
    signals = build_group_physical_signals(data_dir, "train", TARGET).reindex(index)
    frame = _learning_frame(signals, base)
    residual_ratio = (truth - base) / CAPACITY
    train_mask = (
        (index < SELECTION_START)
        & truth.notna()
        & (truth >= 0.10 * CAPACITY)
    )
    selection_mask = (
        (index >= SELECTION_START)
        & (index < LOCKED_START)
        & truth.notna()
    )
    locked_train_mask = (
        (index < LOCKED_START)
        & truth.notna()
        & (truth >= 0.10 * CAPACITY)
    )
    locked_mask = (index >= LOCKED_START) & truth.notna()

    selection_model = _model(seed)
    selection_model.fit(
        frame.loc[train_mask],
        residual_ratio.loc[train_mask],
        eval_set=[(frame.loc[selection_mask], residual_ratio.loc[selection_mask])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = max(100, int(selection_model.best_iteration_))
    selection_correction = selection_model.predict(
        frame.loc[selection_mask], num_iteration=best_iteration
    )
    selection_truth = truth.loc[selection_mask].to_numpy(dtype=float)
    selection_base = base.loc[selection_mask].to_numpy(dtype=float)
    selected_policy, policy_candidates = select_policy(
        selection_truth,
        selection_base,
        selection_correction,
        frame.loc[selection_mask],
    )
    selection_candidate, selection_gate = apply_policy(
        selection_base,
        selection_correction,
        frame.loc[selection_mask],
        selected_policy,
    )

    locked_model = _model(seed, best_iteration)
    locked_model.fit(frame.loc[locked_train_mask], residual_ratio.loc[locked_train_mask])
    locked_correction = locked_model.predict(frame.loc[locked_mask])
    locked_truth = truth.loc[locked_mask].to_numpy(dtype=float)
    locked_base = base.loc[locked_mask].to_numpy(dtype=float)
    locked_candidate, locked_gate = apply_policy(
        locked_base,
        locked_correction,
        frame.loc[locked_mask],
        selected_policy,
    )
    locked_index = index[locked_mask]
    monthly = {}
    for month in range(7, 13):
        month_mask = locked_index.month == month
        monthly[str(month)] = _compare(
            locked_truth[month_mask],
            locked_base[month_mask],
            locked_candidate[month_mask],
        )["delta"]
    locked_comparison = _compare(locked_truth, locked_base, locked_candidate)
    report: dict[str, object] = {
        "method": "group-3 residual correction gated by LDAPS-GFS physical disagreement",
        "hypothesis": (
            "Cross-NWP hub-wind, direction, density-adjusted power and shear disagreement "
            "identify regimes where a small learned correction to the exact current surface "
            "is reliable enough to improve both FICR and 1-NMAE."
        ),
        "source_contract": {
            "raw_data_external_transfer": False,
            "forecast_time_safe": True,
            "nwp_sources": ["official LDAPS", "official GFS"],
        },
        "validation_contract": {
            "selection_train": "2024-01-01 through 2024-04-30 OOF residuals",
            "policy_selection": "2024-05-01 through 2024-06-30",
            "locked_train": "2024-H1 OOF residuals",
            "locked_evaluation": "2024-H2",
        },
        "seed": seed,
        "n_signal_features": int(signals.shape[1]),
        "n_model_features": int(frame.shape[1]),
        "best_iteration": best_iteration,
        "selected_policy": asdict(selected_policy),
        "selection": {
            "changed_rows": int(selection_gate.sum()),
            "changed_ratio": float(selection_gate.mean()),
            "comparison": _compare(
                selection_truth, selection_base, selection_candidate
            ),
            "top_policies": policy_candidates[:10],
        },
        "locked_h2": {
            "changed_rows": int(locked_gate.sum()),
            "changed_ratio": float(locked_gate.mean()),
            "comparison": locked_comparison,
            "monthly_deltas": monthly,
            "positive_months": int(
                sum(value["score"] > 0.0 for value in monthly.values())
            ),
            "day_bootstrap": _bootstrap_days(
                locked_truth,
                locked_base,
                locked_candidate,
                locked_index,
                n_bootstrap,
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    locked_delta = locked_comparison["delta"]
    report["decision"] = (
        "promote_to_test-inference experiment"
        if (
            locked_delta["score"] > 0.0
            and locked_delta["one_minus_nmae"] >= 0.0
            and locked_delta["ficr"] >= 0.0
            and report["locked_h2"]["positive_months"] >= 4
            and report["locked_h2"]["day_bootstrap"]["positive_fraction"] >= 0.75
        )
        else "reject or revise; do not create a submission"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "group3_nwp_disagreement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        artifact_dir / "group3_nwp_disagreement_cache.npz",
        locked_index_ns=locked_index.astype("int64").to_numpy(),
        locked_truth=locked_truth.astype("float32"),
        locked_base=locked_base.astype("float32"),
        locked_candidate=locked_candidate.astype("float32"),
        locked_correction_ratio=locked_correction.astype("float32"),
        locked_gate=locked_gate,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts_final/nwp_disagreement"
    )
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    report = run_experiment(
        Path(args.data_dir),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.artifact_dir),
        args.seed,
        args.n_bootstrap,
    )
    print(
        json.dumps(
            {
                "selection": report["selection"]["comparison"]["delta"],
                "locked_h2": report["locked_h2"]["comparison"]["delta"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
