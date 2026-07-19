from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def apply_policy(prediction: np.ndarray, policy: dict[str, float | str]) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    kind = str(policy["kind"])
    if kind == "threshold_offset":
        output = prediction.copy()
        action = prediction / CAPACITY >= float(policy["minimum_prediction_ratio"])
        output[action] += float(policy["offset"])
    elif kind == "affine":
        output = float(policy["scale"]) * prediction + float(policy["offset"])
    elif kind == "piecewise_offset":
        high = prediction / CAPACITY >= float(policy["breakpoint_ratio"])
        output = prediction + np.where(
            high, float(policy["high_offset"]), float(policy["low_offset"])
        )
    else:
        raise ValueError(f"Unsupported calibration policy: {kind}")
    return np.clip(output, 0.0, CAPACITY)


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def _evaluate(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    before = evaluate_group(truth[mask], base[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _metric_delta(before, after),
    }


def _safe_on_development(q1: dict[str, Any], q2: dict[str, Any]) -> bool:
    return all(
        q1["delta"][name] > 0.0 and q2["delta"][name] > 0.0
        for name in ("score", "one_minus_nmae", "ficr")
    )


def _candidate_record(
    truth: np.ndarray,
    base: np.ndarray,
    index: pd.DatetimeIndex,
    policy: dict[str, float | str],
) -> dict[str, Any] | None:
    candidate = apply_policy(base, policy)
    q1_mask = index < Q2_START
    q2_mask = (index >= Q2_START) & (index < H2_START)
    q1 = _evaluate(truth, base, candidate, q1_mask)
    q2 = _evaluate(truth, base, candidate, q2_mask)
    if not _safe_on_development(q1, q2):
        return None
    movement = np.abs(candidate - base)
    return {
        "policy": policy,
        "development": {"q1": q1, "q2": q2},
        "robust_objective": min(q1["delta"]["score"], q2["delta"]["score"]),
        "mean_development_score_delta": (
            q1["delta"]["score"] + q2["delta"]["score"]
        )
        / 2.0,
        "changed_ratio": float((movement > 1e-9).mean()),
        "mean_absolute_movement_kwh": float(movement.mean()),
    }


def _best(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No policy improved score, 1-NMAE, and FICR in both Q1 and Q2")
    return max(
        records,
        key=lambda row: (
            row["robust_objective"],
            row["mean_development_score_delta"],
            -row["mean_absolute_movement_kwh"],
        ),
    )


def search_threshold_offset(
    truth: np.ndarray, base: np.ndarray, index: pd.DatetimeIndex
) -> dict[str, Any]:
    records = []
    for minimum_ratio in (0.0, 0.10, 0.20, 0.40, 0.50):
        for offset in np.arange(0.0, 1_000.1, 25.0):
            record = _candidate_record(
                truth,
                base,
                index,
                {
                    "kind": "threshold_offset",
                    "minimum_prediction_ratio": minimum_ratio,
                    "offset": float(offset),
                },
            )
            if record is not None:
                records.append(record)
    peak = _best(records)
    # One-standard-error-style safety rule: sacrifice at most 0.00025 local
    # score to affect fewer rows. This is fixed before the locked H2 check.
    near_peak = [
        row
        for row in records
        if row["robust_objective"] >= peak["robust_objective"] - 0.00025
    ]
    selected = min(
        near_peak,
        key=lambda row: (
            row["changed_ratio"],
            row["mean_absolute_movement_kwh"],
            -row["robust_objective"],
        ),
    )
    selected["search_peak_robust_objective"] = peak["robust_objective"]
    selected["safety_tolerance"] = 0.00025
    return selected


def search_affine(
    truth: np.ndarray, base: np.ndarray, index: pd.DatetimeIndex
) -> dict[str, Any]:
    records = []
    for scale in np.arange(0.98, 1.0601, 0.002):
        for offset in np.arange(-200.0, 600.1, 25.0):
            record = _candidate_record(
                truth,
                base,
                index,
                {"kind": "affine", "scale": float(scale), "offset": float(offset)},
            )
            if record is not None:
                records.append(record)
    return _best(records)


def search_piecewise(
    truth: np.ndarray, base: np.ndarray, index: pd.DatetimeIndex
) -> dict[str, Any]:
    records = []
    for breakpoint in (0.20, 0.40, 0.60, 0.80):
        for low_offset in np.arange(-200.0, 800.1, 25.0):
            for high_offset in np.arange(0.0, 1_000.1, 25.0):
                policy = {
                    "kind": "piecewise_offset",
                    "breakpoint_ratio": breakpoint,
                    "low_offset": float(low_offset),
                    "high_offset": float(high_offset),
                }
                movement = np.abs(apply_policy(base, policy) - base).mean()
                if movement > 500.0:
                    continue
                record = _candidate_record(truth, base, index, policy)
                if record is not None:
                    records.append(record)
    return _best(records)


def select_family(families: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    # H2 is not used here. Prefer a simpler calibration when its robust Q1/Q2
    # objective is within 0.00125 of the strongest searched family.
    complexity = {"threshold_offset": 1, "affine": 2, "piecewise_offset": 3}
    peak = max(row["robust_objective"] for row in families.values())
    eligible = {
        name: row
        for name, row in families.items()
        if row["robust_objective"] >= peak - 0.00125
    }
    name = min(
        eligible,
        key=lambda key: (
            complexity[key],
            -eligible[key]["robust_objective"],
            eligible[key]["mean_absolute_movement_kwh"],
        ),
    )
    return name, eligible[name]


def _bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    mask: np.ndarray,
    n_bootstrap: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(20_260_717)
    normalized = index.normalize()
    days = normalized[mask].unique()
    positions = {day: np.flatnonzero(mask & (normalized == day)) for day in days}
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        deltas.append(
            evaluate_group(truth[rows], candidate[rows], CAPACITY).score
            - evaluate_group(truth[rows], base[rows], CAPACITY).score
        )
    values = np.asarray(deltas)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((values > 0.0).mean()),
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def run_experiment(
    driver_cache_path: Path,
    meta_cache_path: Path,
    source_submission_path: Path,
    output_path: Path,
    report_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    driver = np.load(driver_cache_path, allow_pickle=True)
    meta = np.load(meta_cache_path, allow_pickle=True)
    index = pd.DatetimeIndex(pd.to_datetime(driver[f"{TARGET}__valid_index_ns"]))
    meta_index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    if not index.equals(meta_index):
        raise ValueError("Exact driver and meta-gate validation indices do not match")
    truth = driver[f"{TARGET}__valid_truth"].astype(float)
    base = meta["valid_candidate"].astype(float)

    family_winners = {
        "threshold_offset": search_threshold_offset(truth, base, index),
        "affine": search_affine(truth, base, index),
        "piecewise_offset": search_piecewise(truth, base, index),
    }
    selected_family, selected = select_family(family_winners)
    h2 = index >= H2_START
    selected_prediction = apply_policy(base, selected["policy"])
    for row in family_winners.values():
        row["locked_h2"] = _evaluate(
            truth, base, apply_policy(base, row["policy"]), h2
        )

    monthly = {}
    positive_months = 0
    for month in range(7, 13):
        month_mask = h2 & (index.month == month)
        evaluation = _evaluate(truth, base, selected_prediction, month_mask)
        monthly[str(month)] = evaluation["delta"]
        positive_months += evaluation["delta"]["score"] > 0.0
    bootstrap = _bootstrap_days(
        truth, base, selected_prediction, index, h2, n_bootstrap
    )
    locked_delta = selected["locked_h2"]["delta"]

    source = pd.read_csv(source_submission_path, encoding="utf-8-sig")
    expected_columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
        TARGET,
    ]
    if source.columns.tolist() != expected_columns:
        raise ValueError("Source submission schema does not match the competition schema")
    test_index = pd.DatetimeIndex(pd.to_datetime(source["forecast_kst_dtm"]))
    meta_test_index = pd.DatetimeIndex(pd.to_datetime(meta["test_index_ns"]))
    if not test_index.equals(meta_test_index):
        raise ValueError("Source submission and meta-gate test indices do not match")
    cached_test = meta["test_candidate"].astype(float)
    source_test = source[TARGET].to_numpy(dtype=float)
    parity_error = np.abs(cached_test - source_test)
    if parity_error.max() > 0.01:
        raise ValueError("Source submission no longer matches the frozen meta-gate cache")
    calibrated_test = apply_policy(source_test, selected["policy"])
    output = source.copy()
    output[TARGET] = calibrated_test

    schema_ok = output.columns.tolist() == expected_columns and len(output) == len(source)
    finite_ok = bool(np.isfinite(output[list(CAPACITY_KWH)].to_numpy(dtype=float)).all())
    bounds_ok = all(
        bool(output[target].between(0.0, capacity).all())
        for target, capacity in CAPACITY_KWH.items()
    )
    groups_1_2_unchanged = bool(
        np.array_equal(output["kpx_group_1"], source["kpx_group_1"])
        and np.array_equal(output["kpx_group_2"], source["kpx_group_2"])
    )
    promote = bool(
        locked_delta["score"] > 0.0
        and locked_delta["one_minus_nmae"] > 0.0
        and locked_delta["ficr"] > 0.0
        and positive_months >= 4
        and bootstrap["positive_fraction"] >= 0.85
        and schema_ok
        and finite_ok
        and bounds_ok
        and groups_1_2_unchanged
    )
    if promote:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")

    movement = calibrated_test - source_test
    report: dict[str, Any] = {
        "method": "threshold-aware affine and piecewise calibration on exact meta-gate OOF",
        "selection_protocol": {
            "development": "select on Q1/Q2 consistency only",
            "locked": "H2 is evaluated after family and parameters are frozen",
            "required_development_components": ["score", "one_minus_nmae", "ficr"],
            "family_simplicity_tolerance": 0.00125,
            "selected_family": selected_family,
            "selected_policy": selected["policy"],
        },
        "family_winners": family_winners,
        "locked_h2_monthly_deltas": monthly,
        "locked_h2_positive_months": positive_months,
        "locked_h2_day_bootstrap": bootstrap,
        "promotion": {
            "promoted": promote,
            "rules": {
                "locked_all_components_positive": True,
                "minimum_positive_months": 4,
                "minimum_bootstrap_positive_fraction": 0.85,
            },
            "estimated_public_score_delta_from_locked_macro_scaling": (
                locked_delta["score"] / 3.0
            ),
        },
        "test_output": {
            "source": str(source_submission_path),
            "output": str(output_path) if promote else None,
            "rows": len(output),
            "changed_rows": int((np.abs(movement) > 1e-9).sum()),
            "changed_ratio": float((np.abs(movement) > 1e-9).mean()),
            "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
            "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "max_absolute_movement_kwh": float(np.abs(movement).max()),
            "source_cache_parity_max_error_kwh": float(parity_error.max()),
            "groups_1_2_unchanged": groups_1_2_unchanged,
            "schema_ok": schema_ok,
            "finite_ok": finite_ok,
            "bounds_ok": bounds_ok,
        },
        "decision": (
            "Promote one controlled submission candidate: the simple threshold-offset policy "
            "was selected without H2, then improved locked score, 1-NMAE, and FICR."
            if promote
            else "Reject calibration: the predeclared locked promotion rules were not met."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--source-submission",
        default="submissions/blend_best_crossg3_traj_meta25_p55.csv",
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_meta_g3_thr10_off575.csv"
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/threshold_calibration/threshold_calibration_report.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run_experiment(
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.source_submission),
        Path(args.output),
        Path(args.report),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
