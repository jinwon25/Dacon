"""Locked validation on train issue cycles most similar to the 2025 test NWP.

This diagnostic deliberately uses only covariates available before submission:
the train/test GFS issue-cycle summaries and calendar timestamps.  It never
reads 2025 targets or SCADA data.  The analogue set is frozen before looking at
validation labels (nearest 25 percent of 2024 H2 issue cycles).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import assign_issue_blocks, load_issue_times
from experiments.spatiotemporal_consensus_promotion import _rolling_finesweep_base
from src.metrics import CAPACITY_KWH, evaluate_competition


ROOT = Path(__file__).resolve().parents[1]
TARGETS = list(CAPACITY_KWH)
H2_START = pd.Timestamp("2024-07-01")
H2_END = pd.Timestamp("2025-01-01")
ANALOGUE_QUANTILE = 0.25
BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 20260718
GFS_FEATURES = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "heightAboveGround_2_2t",
    "surface_0_gust",
)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _load_exact() -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(ROOT / "artifacts_final/lineage/exact_driver_oof.npz")
    index = pd.DatetimeIndex(pd.to_datetime(cache["kpx_group_1__valid_index_ns"]))
    truth = np.column_stack(
        [cache[f"{target}__valid_truth"].astype(float) for target in TARGETS]
    )
    base = np.column_stack(
        [cache[f"{target}__exact_base"].astype(float) for target in TARGETS]
    )
    if any(len(cache[f"{target}__valid_index_ns"]) != len(index) for target in TARGETS):
        raise ValueError("Exact OOF group indexes are not aligned")
    return index, truth, {"exact_driver_base": base}


def _load_neural(path: Path, exact_index: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    cache = np.load(path, allow_pickle=False)
    timestamps = pd.DatetimeIndex(pd.to_datetime(cache["timestamps_ns"]))
    prediction = cache["prediction"].astype(float)
    if prediction.shape != (len(timestamps), len(TARGETS)):
        raise ValueError(f"Unexpected neural cache shape: {path}")
    positions = timestamps.get_indexer(exact_index)
    keep = positions >= 0
    output = np.full((len(exact_index), len(TARGETS)), np.nan, dtype=float)
    output[keep] = prediction[positions[keep]] * np.asarray(
        [CAPACITY_KWH[target] for target in TARGETS]
    )
    return output, int((~keep).sum())


def _cycle_features(path: Path, start: str, end: str) -> pd.DataFrame:
    columns = ["forecast_kst_dtm", "data_available_kst_dtm", "grid_id", *GFS_FEATURES]
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(frame["data_available_kst_dtm"])
    frame = frame[
        (frame["grid_id"] == 5)
        & (frame["forecast_kst_dtm"] >= pd.Timestamp(start))
        & (frame["forecast_kst_dtm"] < pd.Timestamp(end))
    ].copy()
    if frame.empty:
        raise ValueError(f"No center-grid GFS rows in {path}")
    frame["speed10"] = np.hypot(
        frame["heightAboveGround_10_10u"], frame["heightAboveGround_10_10v"]
    )
    frame["speed100"] = np.hypot(
        frame["heightAboveGround_100_100u"], frame["heightAboveGround_100_100v"]
    )
    # The feature contract is fixed up front: location-independent weather
    # level and dispersion summaries plus issue-calendar phase.
    weather = ["speed10", "speed100", "surface_0_gust", "heightAboveGround_2_2t"]
    grouped = frame.groupby("data_available_kst_dtm", sort=True)
    records: list[dict[str, Any]] = []
    for issue, values in grouped:
        record: dict[str, Any] = {
            "issue": issue,
            "representative": values["forecast_kst_dtm"].median(),
        }
        for name in weather:
            array = values[name].to_numpy(dtype=float)
            record[f"{name}_mean"] = float(np.nanmean(array))
            record[f"{name}_std"] = float(np.nanstd(array))
            record[f"{name}_p10"] = float(np.nanpercentile(array, 10))
            record[f"{name}_p90"] = float(np.nanpercentile(array, 90))
        issue_time = pd.Timestamp(issue)
        phase = issue_time.dayofyear + (issue_time.hour / 24.0)
        record["month_sin"] = float(np.sin(2.0 * np.pi * issue_time.month / 12.0))
        record["month_cos"] = float(np.cos(2.0 * np.pi * issue_time.month / 12.0))
        record["doy_sin"] = float(np.sin(2.0 * np.pi * phase / 365.25))
        record["doy_cos"] = float(np.cos(2.0 * np.pi * phase / 365.25))
        record["issue_hour_sin"] = float(np.sin(2.0 * np.pi * issue_time.hour / 24.0))
        record["issue_hour_cos"] = float(np.cos(2.0 * np.pi * issue_time.hour / 24.0))
        records.append(record)
    result = pd.DataFrame.from_records(records).set_index("issue").sort_index()
    if result.isna().any().any():
        raise ValueError(f"NaN in cycle features: {path}")
    return result


def _nearest_analogue_cycles(train_cycles: pd.DataFrame, test_cycles: pd.DataFrame) -> tuple[pd.Index, dict[str, Any]]:
    train_h2 = train_cycles[
        (train_cycles["representative"] >= H2_START)
        & (train_cycles["representative"] < H2_END)
    ]
    feature_cols = [column for column in train_cycles.columns if column not in ("representative",)]
    train_values = train_h2[feature_cols].to_numpy(dtype=float)
    test_values = test_cycles[feature_cols].to_numpy(dtype=float)
    center = train_cycles[feature_cols].to_numpy(dtype=float).mean(axis=0)
    scale = train_cycles[feature_cols].to_numpy(dtype=float).std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_values = (train_values - center) / scale
    test_values = (test_values - center) / scale
    nearest_distance = np.sqrt(
        ((train_values[:, None, :] - test_values[None, :, :]) ** 2).mean(axis=2)
    ).min(axis=1)
    threshold = float(np.quantile(nearest_distance, ANALOGUE_QUANTILE))
    selected = train_h2.index[nearest_distance <= threshold + 1e-12]
    details = {
        "method": "nearest_test_issue_cycle",
        "feature_columns": feature_cols,
        "standardization": "train_2024_cycle_mean_std",
        "candidate_train_h2_cycles": int(len(train_h2)),
        "test_cycles": int(len(test_cycles)),
        "selection_quantile": ANALOGUE_QUANTILE,
        "selection_distance_threshold": threshold,
        "distance_quantiles": {
            str(q): float(np.quantile(nearest_distance, q)) for q in (0.0, 0.25, 0.50, 0.75, 1.0)
        },
        "selected_cycles": int(len(selected)),
        "selected_cycle_ids": [pd.Timestamp(item).isoformat() for item in selected],
    }
    # A pre-declared wider set is reported only as sensitivity; it is not used
    # to choose a model or to alter the primary analogue25 decision.
    for quantile in (0.50,):
        details[f"sensitivity_q{int(quantile * 100)}_cycles"] = int(
            np.sum(nearest_distance <= np.quantile(nearest_distance, quantile) + 1e-12)
        )
    return selected, details


def _metrics(truth: np.ndarray, prediction: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    if not np.asarray(rows, dtype=bool).any():
        raise ValueError("Empty validation subset")
    return evaluate_competition(
        {target: truth[rows, i] for i, target in enumerate(TARGETS)},
        {target: prediction[rows, i] for i, target in enumerate(TARGETS)},
    )


def _delta(metrics: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key] - reference[key])
        for key in ("score", "one_minus_nmae", "ficr")
    }


def _issue_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    rows: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    _, strata = assign_issue_blocks(index, issue_times)
    issue_values = np.asarray(issue_times)
    strata_issues = {
        str(stratum): np.unique(issue_values[rows & (strata == stratum)])
        for stratum in sorted(set(strata[rows]))
    }
    positions = {
        (str(stratum), issue): np.flatnonzero(
            rows & (strata == stratum) & (issue_values == issue)
        )
        for stratum, issues in strata_issues.items()
        for issue in issues
    }
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(n_bootstrap, dtype=float) for name in ("score", "one_minus_nmae", "ficr")}
    for iteration in range(n_bootstrap):
        sampled_rows: list[np.ndarray] = []
        for stratum, issues in strata_issues.items():
            sampled = rng.choice(issues, size=len(issues), replace=True)
            sampled_rows.extend(positions[(stratum, issue)] for issue in sampled)
        selected = np.concatenate(sampled_rows)
        candidate_metric = _metrics(truth, candidate, np.isin(np.arange(len(index)), selected))
        reference_metric = _metrics(truth, reference, np.isin(np.arange(len(index)), selected))
        for name in deltas:
            deltas[name][iteration] = candidate_metric[name] - reference_metric[name]
    return {
        "n_bootstrap": n_bootstrap,
        "strata_issue_cycles": {key: int(len(value)) for key, value in strata_issues.items()},
        **{
            name: {
                "positive_fraction": float(np.mean(values > 0.0)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
            }
            for name, values in deltas.items()
        },
    }


def run() -> dict[str, Any]:
    exact_index, exact_truth, _ = _load_exact()
    cache = np.load(ROOT / "artifacts_final/lineage/exact_driver_oof.npz")
    base = np.column_stack([cache[f"{target}__exact_base"].astype(float) for target in TARGETS])

    # Rebuild the locked p=.545 / alpha=.50 finesweep from retained OOF lineage.
    _, fine_index, _, fine_group3 = _rolling_finesweep_base(
        ROOT / "data/train/train_labels.csv",
        ROOT / "artifacts_final/lineage/exact_driver_oof.npz",
    )
    if not fine_index.equals(exact_index):
        raise ValueError("Finesweep and exact OOF indexes differ")
    fine = base.copy()
    fine[:, 2] = fine_group3

    structural = np.load(ROOT / "artifacts_final/structural_20260718/locked_predictions.npz")
    structural_index = pd.DatetimeIndex(pd.to_datetime(structural["index_ns"]))
    if not structural_index.equals(exact_index):
        raise ValueError("Structural locked cache is not on exact OOF index")
    trajectory = base.copy()
    trajectory[:, 2] = structural["locked_candidate"].astype(float)

    neural_paths = {
        "spatiotemporal_seed17": ROOT / "artifacts_final/spatiotemporal/validation_predictions.npz",
        "spatiotemporal_seed29": ROOT / "artifacts_final/spatiotemporal_seed29/validation_predictions.npz",
    }
    neural: dict[str, np.ndarray] = {}
    neural_missing: dict[str, int] = {}
    for name, path in neural_paths.items():
        neural[name], neural_missing[name] = _load_neural(path, exact_index)
    neural_stack = np.stack(list(neural.values()))
    valid_count = np.sum(np.isfinite(neural_stack), axis=0)
    neural_ensemble = np.divide(
        np.nansum(neural_stack, axis=0),
        valid_count,
        out=np.full(neural_stack.shape[1:], np.nan, dtype=float),
        where=valid_count > 0,
    )

    # All models are compared on one common exact-index intersection.  The
    # neural caches omit one boundary timestamp; no labels are imputed.
    common = np.isfinite(neural_ensemble).all(axis=1)
    common_index = exact_index[common]
    truth = exact_truth[common]
    issue_times = load_issue_times(ROOT / "data/train/gfs_train.csv", exact_index)[common]
    model_predictions = {
        "incumbent_finesweep_p545_a50": fine[common],
        "exact_driver_base_reference": base[common],
        "corrected_cycle_trajectory_locked": trajectory[common],
        "spatiotemporal_seed17": neural["spatiotemporal_seed17"][common],
        "spatiotemporal_seed29": neural["spatiotemporal_seed29"][common],
        "spatiotemporal_seed17_29_mean": neural_ensemble[common],
    }
    for name, prediction in model_predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite prediction after common-index alignment: {name}")

    train_cycles = _cycle_features(ROOT / "data/train/gfs_train.csv", "2024-01-01", "2025-01-01")
    test_cycles = _cycle_features(ROOT / "data/test/gfs_test.csv", "2025-01-01", "2026-01-01")
    selected_cycles, selection = _nearest_analogue_cycles(train_cycles, test_cycles)
    selected_set = set(selected_cycles)
    analogue25 = np.asarray([issue in selected_set for issue in issue_times], dtype=bool)
    h2 = (common_index >= H2_START) & (common_index < H2_END)
    analogue25 &= h2
    # Fixed q=.50 sensitivity, still selected without labels.
    train_h2 = train_cycles[(train_cycles["representative"] >= H2_START) & (train_cycles["representative"] < H2_END)]
    feature_cols = [column for column in train_cycles.columns if column != "representative"]
    # Recompute the declared q=.50 membership using the same standardized space.
    center = train_cycles[feature_cols].to_numpy(float).mean(axis=0)
    scale = train_cycles[feature_cols].to_numpy(float).std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_values = (train_h2[feature_cols].to_numpy(float) - center) / scale
    test_values = (test_cycles[feature_cols].to_numpy(float) - center) / scale
    distances = np.sqrt(((train_values[:, None, :] - test_values[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
    q50_set = set(train_h2.index[distances <= np.quantile(distances, 0.50) + 1e-12])
    analogue50 = np.asarray([issue in q50_set for issue in issue_times], dtype=bool) & h2

    subsets = {
        "general_h2": h2,
        "test_analogue25_h2": analogue25,
        "test_analogue50_sensitivity_h2": analogue50,
    }
    subset_summary = {
        name: {
            "rows": int(mask.sum()),
            "issue_cycles": int(pd.Index(issue_times[mask]).nunique()),
            "start": common_index[mask].min().isoformat() if mask.any() else None,
            "end": common_index[mask].max().isoformat() if mask.any() else None,
        }
        for name, mask in subsets.items()
    }
    reference = model_predictions["incumbent_finesweep_p545_a50"]
    model_results: dict[str, Any] = {}
    for model_name, prediction in model_predictions.items():
        model_results[model_name] = {}
        for subset_name, mask in subsets.items():
            metrics = _metrics(truth, prediction, mask)
            entry: dict[str, Any] = {"metrics": metrics}
            if model_name != "incumbent_finesweep_p545_a50":
                entry["delta_vs_incumbent"] = _delta(
                    metrics, _metrics(truth, reference, mask)
                )
                entry["issue_cycle_bootstrap_vs_incumbent"] = _issue_bootstrap(
                    truth, reference, prediction, common_index, issue_times, mask
                )
            model_results[model_name][subset_name] = entry

    result: dict[str, Any] = {
        "schema_version": "test_analogue_validation.v1",
        "created_at": "2026-07-18",
        "purpose": "covariate-only 2025-test analogue locked validation; no public score or 2025 target/SCADA",
        "lineage": {
            "exact_driver_cache": "artifacts_final/lineage/exact_driver_oof.npz",
            "incumbent": "rolling finesweep p=.545 alpha=.50; Q1/H1-fitted probabilities",
            "trajectory_member": "artifacts_final/structural_20260718/locked_predictions.npz",
            "neural_members": [str(path.relative_to(ROOT)) for path in neural_paths.values()],
            "evaluation_index_rows": int(len(common_index)),
            "exact_oof_rows": int(len(exact_index)),
            "neural_boundary_rows_dropped": neural_missing,
        },
        "analogue_selection": selection,
        "subset_summary": subset_summary,
        "models": model_results,
        "guardrails": [
            "Only 2024 train GFS and 2025 test GFS covariates/calendar were used for cycle selection.",
            "No 2025 targets, SCADA, public leaderboard score, or submission file was read or written.",
            "Analogue25 was fixed at nearest-distance quantile 0.25 before metric evaluation.",
            "Analogue50 is sensitivity only; no model or hyperparameter is selected from it.",
            "Issue-cycle bootstrap resamples complete NWP issue cycles stratified by representative season.",
        ],
    }
    return _to_builtin(result)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Test-analogue locked validation (2026-07-18)",
        "",
        "2025 test GFS/time covariates select a fixed nearest-cycle subset of 2024 H2. No test target/SCADA, public score, or submission was used.",
        "",
        f"- Evaluation index: {report['lineage']['evaluation_index_rows']} / {report['lineage']['exact_oof_rows']} exact OOF rows (one neural boundary row dropped).",
        f"- Analogue rule: nearest 2025 test issue cycle, train-2024 standardised cycle summaries, bottom {report['analogue_selection']['selection_quantile']:.0%} distance quantile.",
        f"- Selected cycles: {report['analogue_selection']['selected_cycles']} / {report['analogue_selection']['candidate_train_h2_cycles']}; threshold {report['analogue_selection']['selection_distance_threshold']:.4f}.",
        "",
        "## Locked subset coverage",
        "",
        "| subset | rows | issue cycles |",
        "|---|---:|---:|",
    ]
    for name, summary in report["subset_summary"].items():
        lines.append(f"| {name} | {summary['rows']} | {summary['issue_cycles']} |")
    lines += ["", "## Metrics and differences vs incumbent finesweep", "", "| model | subset | score | 1-NMAE | FiCR | Δ score | Δ 1-NMAE | Δ FiCR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, subsets in report["models"].items():
        for subset, entry in subsets.items():
            m = entry["metrics"]
            d = entry.get("delta_vs_incumbent", {"score": 0.0, "one_minus_nmae": 0.0, "ficr": 0.0})
            lines.append(f"| {model} | {subset} | {m['score']:.6f} | {m['one_minus_nmae']:.6f} | {m['ficr']:.6f} | {d['score']:+.6f} | {d['one_minus_nmae']:+.6f} | {d['ficr']:+.6f} |")
    lines += ["", "## Issue-cycle bootstrap direction (500 resamples)", "", "Positive fraction is P(candidate − incumbent > 0); q05/q95 show cycle uncertainty.", "", "| model | subset | score P+ / median [q05,q95] | 1-NMAE P+ / median [q05,q95] | FiCR P+ / median [q05,q95] |", "|---|---|---|---|---|"]
    for model, subsets in report["models"].items():
        if model == "incumbent_finesweep_p545_a50":
            continue
        for subset, entry in subsets.items():
            b = entry["issue_cycle_bootstrap_vs_incumbent"]
            cells = []
            for key in ("score", "one_minus_nmae", "ficr"):
                x = b[key]
                cells.append(f"{x['positive_fraction']:.3f} / {x['median']:+.5f} [{x['q05']:+.5f},{x['q95']:+.5f}]")
            lines.append(f"| {model} | {subset} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation", "", "The analogue subset is a covariate-shift diagnostic, not a leaderboard objective. A candidate is not promoted from one analogue score: require directionally positive score, 1-NMAE and FiCR with issue-cycle bootstrap support, and retain the incumbent if this fails.", ""]
    return "\n".join(lines)


def main() -> None:
    report = run()
    out_dir = ROOT / "artifacts_final/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_analogue_validation_20260718.json"
    md_path = out_dir / "test_analogue_validation_20260718.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
