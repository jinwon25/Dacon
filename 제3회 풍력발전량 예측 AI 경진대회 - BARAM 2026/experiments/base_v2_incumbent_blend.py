from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.exact_oof_meta_gate import (
    actionable_mask,
    apply_meta_gate,
    settlement_benefit_labels,
)
from experiments.exact_oof_meta_gate_sweep import (
    H2_START,
    Q2_START,
    _prepare_validation,
    fit_probabilities,
)
from experiments.nested_quantile_base import (
    _competition_delta,
    _issue_bootstrap,
    _ordered_issue_seasons,
)
from src.metrics import CAPACITY_KWH, evaluate_competition


FINE_THRESHOLD = 0.545
FINE_ALPHA = 0.50


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def choose_weight(
    truth: dict[str, np.ndarray],
    incumbent: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    base = evaluate_competition(truth, incumbent)
    records: list[dict[str, Any]] = []
    for weight in weights:
        prediction = {
            group: (1.0 - float(weight)) * incumbent[group]
            + float(weight) * member[group]
            for group in CAPACITY_KWH
        }
        metric = evaluate_competition(truth, prediction)
        records.append(
            {
                "weight": float(weight),
                "metric": metric,
                "delta": {
                    key: float(metric[key] - base[key])
                    for key in ("score", "one_minus_nmae", "ficr")
                },
            }
        )
    eligible = [
        record
        for record in records
        if record["delta"]["one_minus_nmae"] >= 0.0
        and record["delta"]["ficr"] >= 0.0
    ]
    if not eligible:
        return 0.0, records
    selected = max(
        eligible,
        key=lambda record: (
            record["metric"]["score"],
            -record["weight"],
        ),
    )
    return float(selected["weight"]), records


def _align(
    left_index: pd.DatetimeIndex,
    right_index: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    common = left_index.intersection(right_index)
    if common.empty:
        raise ValueError("Nested Base v2 and incumbent OOF have no common timestamps")
    return common, left_index.get_indexer(common), right_index.get_indexer(common)


def run(
    *,
    base_run: Path,
    labels_path: Path,
    driver_path: Path,
    fine_cache_path: Path,
    gfs_train_path: Path,
    incumbent_submission_path: Path,
    output_submission_path: Path,
    report_path: Path,
    weights: np.ndarray,
    n_bootstrap: int,
    write_submission_if_qualified: bool,
) -> dict[str, Any]:
    nested_path = base_run / "predictions.npz"
    nested = _load_npz(nested_path)
    driver = _load_npz(driver_path)
    fine = _load_npz(fine_cache_path)
    nested_index = pd.DatetimeIndex(pd.to_datetime(nested["index_ns"]))
    incumbent_index = pd.DatetimeIndex(
        pd.to_datetime(driver["kpx_group_1__valid_index_ns"])
    )
    fine_index = pd.DatetimeIndex(pd.to_datetime(fine["valid_index_ns"]))
    if not incumbent_index.equals(fine_index):
        raise ValueError("Fine meta-gate and incumbent OOF timestamps differ")

    (
        _labels,
        prepared_index,
        truth_group3,
        group1,
        group2,
        _base_group3,
        member_group3,
        current_group3,
        meta_features,
        action,
    ) = _prepare_validation(labels_path, driver_path)
    if not prepared_index.equals(incumbent_index):
        raise ValueError("Prepared incumbent timestamps differ from driver cache")
    q1 = incumbent_index < Q2_START
    q2_probability, _ = fit_probabilities(
        meta_features,
        settlement_benefit_labels(
            truth_group3,
            current_group3,
            member_group3,
            q1,
        ),
        q1 & action,
    )
    development_group3, _ = apply_meta_gate(
        current_group3,
        member_group3,
        actionable_mask(group1, group2, _base_group3, member_group3)
        & (truth_group3 >= 0.10 * CAPACITY_KWH["kpx_group_3"]),
        q2_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    locked_group3 = fine["valid_candidate"].astype(float)

    common, nested_rows, incumbent_rows = _align(
        nested_index,
        incumbent_index,
    )
    truth: dict[str, np.ndarray] = {}
    nested_prediction: dict[str, np.ndarray] = {}
    development_incumbent: dict[str, np.ndarray] = {}
    locked_incumbent: dict[str, np.ndarray] = {}
    for group in CAPACITY_KWH:
        nested_truth = nested[f"{group}__truth"][nested_rows].astype(float)
        driver_truth = driver[f"{group}__valid_truth"][incumbent_rows].astype(float)
        if not np.allclose(nested_truth, driver_truth, rtol=0.0, atol=1e-5):
            raise ValueError(f"Nested and incumbent truth differ for {group}")
        truth[group] = nested_truth
        nested_prediction[group] = nested[f"{group}__candidate"][
            nested_rows
        ].astype(float)
        if group == "kpx_group_3":
            development_incumbent[group] = development_group3[
                incumbent_rows
            ].astype(float)
            locked_incumbent[group] = locked_group3[incumbent_rows]
        else:
            exact = driver[f"{group}__exact_base"][incumbent_rows].astype(float)
            development_incumbent[group] = exact
            locked_incumbent[group] = exact

    development = (
        (common >= Q2_START)
        & (common < H2_START)
    )
    # The nested seasonal OOF ends with a single 2024-12-01 boundary row.
    # Keep only complete locked calendar months so that one row cannot become
    # a spurious "December" robustness gate.
    locked = (common >= H2_START) & (common < pd.Timestamp("2024-12-01"))
    if not development.any() or not locked.any():
        raise ValueError("Common OOF does not cover both Q2 development and H2 locked periods")
    selected_weight, development_records = choose_weight(
        {group: values[development] for group, values in truth.items()},
        {
            group: values[development]
            for group, values in development_incumbent.items()
        },
        {
            group: values[development]
            for group, values in nested_prediction.items()
        },
        weights,
    )
    locked_candidate = {
        group: (1.0 - selected_weight) * locked_incumbent[group][locked]
        + selected_weight * nested_prediction[group][locked]
        for group in CAPACITY_KWH
    }
    locked_truth = {group: values[locked] for group, values in truth.items()}
    locked_reference = {
        group: values[locked] for group, values in locked_incumbent.items()
    }
    locked_result = _competition_delta(
        locked_truth,
        locked_reference,
        locked_candidate,
    )
    locked_index = common[locked]
    issue_times = load_issue_times(gfs_train_path, locked_index)
    seasons, _ = _ordered_issue_seasons(locked_index, issue_times)
    bootstrap = _issue_bootstrap(
        locked_truth,
        locked_reference,
        locked_candidate,
        issue_times,
        seasons,
        n_bootstrap=n_bootstrap,
        seed=20_260_725,
    )
    monthly: dict[str, Any] = {}
    for month in dict.fromkeys(locked_index.to_period("M").astype(str)):
        rows = locked_index.to_period("M").astype(str) == month
        monthly[month] = _competition_delta(
            {group: values[rows] for group, values in locked_truth.items()},
            {group: values[rows] for group, values in locked_reference.items()},
            {group: values[rows] for group, values in locked_candidate.items()},
        )
    incumbent_submission = pd.read_csv(
        incumbent_submission_path,
        encoding="utf-8-sig",
    )
    test_index = pd.DatetimeIndex(pd.to_datetime(nested["test_index_ns"]))
    submission_index = pd.DatetimeIndex(
        pd.to_datetime(incumbent_submission["forecast_kst_dtm"])
    )
    if not test_index.equals(submission_index):
        raise ValueError("Nested and incumbent submission timestamps differ")
    test_candidate: dict[str, np.ndarray] = {}
    normalized_movement: list[np.ndarray] = []
    test_movement: dict[str, Any] = {}
    changed_target_cells = 0
    total_target_cells = 0
    for group, capacity in CAPACITY_KWH.items():
        incumbent_values = incumbent_submission[group].to_numpy(dtype=float)
        candidate_values = np.clip(
            (1.0 - selected_weight) * incumbent_values
            + selected_weight * nested[f"{group}__test"].astype(float),
            0.0,
            capacity,
        )
        test_candidate[group] = candidate_values
        movement = candidate_values - incumbent_values
        absolute = np.abs(movement)
        normalized_movement.append(absolute / capacity)
        changed = absolute > 1e-6
        changed_target_cells += int(changed.sum())
        total_target_cells += len(changed)
        test_movement[group] = {
            "mean_absolute_kwh": float(np.mean(absolute)),
            "p95_absolute_kwh": float(np.quantile(absolute, 0.95)),
            "maximum_absolute_kwh": float(np.max(absolute)),
            "changed_rows": int(changed.sum()),
            "changed_ratio": float(changed.mean()),
            "p95_movement_ratio": float(np.quantile(absolute / capacity, 0.95)),
        }
    movement_vector = np.concatenate(normalized_movement)
    movement_summary = {
        "changed_target_cells": changed_target_cells,
        "total_target_cells": total_target_cells,
        "changed_target_cell_ratio": float(
            changed_target_cells / total_target_cells
        ),
        "p95_movement_ratio": float(np.quantile(movement_vector, 0.95)),
        "by_group": test_movement,
    }
    gates = {
        "selected_nonzero_weight": selected_weight > 0.0,
        "locked_score_positive": locked_result["delta"]["score"] > 0.0,
        "locked_one_minus_nmae_nonnegative": locked_result["delta"][
            "one_minus_nmae"
        ]
        >= 0.0,
        "locked_ficr_nonnegative": locked_result["delta"]["ficr"] >= 0.0,
        "worst_locked_month_nonnegative": min(
            result["delta"]["score"] for result in monthly.values()
        )
        >= 0.0,
        "bootstrap_q05_nonnegative": bootstrap["q05"] >= 0.0,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.90,
        "changed_target_cell_ratio_bounded": (
            movement_summary["changed_target_cell_ratio"] <= 0.25
        ),
        "p95_movement_ratio_bounded": (
            movement_summary["p95_movement_ratio"] <= 0.015
        ),
    }
    qualified = bool(all(gates.values()))
    exploratory_gates = {
        "selected_nonzero_weight": selected_weight > 0.0,
        "locked_score_minimum": locked_result["delta"]["score"] >= 0.00015,
        "locked_one_minus_nmae_tolerable": locked_result["delta"][
            "one_minus_nmae"
        ]
        >= -0.00035,
        "locked_ficr_tolerable": locked_result["delta"]["ficr"] >= -0.00035,
        "worst_locked_month_tolerable": min(
            result["delta"]["score"] for result in monthly.values()
        )
        >= -0.00035,
        "bootstrap_q05_tolerable": bootstrap["q05"] >= -0.00025,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.80,
        "changed_target_cell_ratio_bounded": (
            movement_summary["changed_target_cell_ratio"] <= 0.50
        ),
        "p95_movement_ratio_bounded": (
            movement_summary["p95_movement_ratio"] <= 0.025
        ),
    }
    decision_tier = (
        "candidate"
        if qualified
        else "exploratory"
        if all(exploratory_gates.values())
        else "rejected"
    )

    submission = None
    if qualified and write_submission_if_qualified:
        output = incumbent_submission.copy()
        for group in CAPACITY_KWH:
            output[group] = test_candidate[group]
        output_submission_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(
            output_submission_path,
            index=False,
            encoding="utf-8-sig",
        )
        submission = {
            "path": output_submission_path.as_posix(),
            "rows": len(output),
            "sha256": hashlib.sha256(
                output_submission_path.read_bytes()
            ).hexdigest(),
        }

    report: dict[str, Any] = {
        "method": "Q1-trained incumbent meta gate, Q2 blend selection, H2 locked evaluation",
        "sources": {
            "nested_base": nested_path.as_posix(),
            "incumbent_oof": driver_path.as_posix(),
            "fine_meta_cache": fine_cache_path.as_posix(),
            "incumbent_submission": incumbent_submission_path.as_posix(),
        },
        "common_oof": {
            "rows": len(common),
            "start": str(common.min()),
            "end": str(common.max()),
            "development_rows": int(development.sum()),
            "locked_rows": int(locked.sum()),
        },
        "development": {
            "period": "2024-Q2",
            "weights": development_records,
            "selected_weight": selected_weight,
        },
        "locked_h2": locked_result,
        "locked_monthly": monthly,
        "issue_block_bootstrap": bootstrap,
        "test_movement": movement_summary,
        "promotion_gates": gates,
        "exploratory_gates": exploratory_gates,
        "decision_tier": decision_tier,
        "qualified": qualified,
        "submission_requested": write_submission_if_qualified,
        "submission": submission,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run",
        default="artifacts_final/base_v2/nested_quantile_lgbm",
    )
    parser.add_argument(
        "--labels",
        default="data/train/train_labels.csv",
    )
    parser.add_argument(
        "--driver-cache",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    parser.add_argument(
        "--fine-cache",
        default="artifacts_final/meta_gate/fine_sweep_rolling_oof.npz",
    )
    parser.add_argument(
        "--gfs-train",
        default="data/train/gfs_train.csv",
    )
    parser.add_argument(
        "--incumbent-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="submissions/base_v2_incumbent_blend.csv",
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/base_v2/incumbent_blend_report.json",
    )
    parser.add_argument("--maximum-weight", type=float, default=0.50)
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.weight_step <= args.maximum_weight <= 1.0:
        raise ValueError("weight grid must satisfy 0 < step <= maximum <= 1")
    weights = np.arange(
        0.0,
        args.maximum_weight + args.weight_step / 2.0,
        args.weight_step,
    )
    report = run(
        base_run=Path(args.base_run),
        labels_path=Path(args.labels),
        driver_path=Path(args.driver_cache),
        fine_cache_path=Path(args.fine_cache),
        gfs_train_path=Path(args.gfs_train),
        incumbent_submission_path=Path(args.incumbent_submission),
        output_submission_path=Path(args.output_submission),
        report_path=Path(args.report),
        weights=weights,
        n_bootstrap=args.n_bootstrap,
        write_submission_if_qualified=args.write_submission_if_qualified,
    )
    print(
        json.dumps(
            {
                "selected_weight": report["development"]["selected_weight"],
                "locked_h2": report["locked_h2"],
                "issue_block_bootstrap": report["issue_block_bootstrap"],
                "promotion_gates": report["promotion_gates"],
                "qualified": report["qualified"],
                "submission": report["submission"],
                "report": args.report,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
