from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.kma_base_v2_local_overlay import (
    H2_START,
    LOCKED_END,
    Q2_START,
    OverlayPolicy,
    _align,
    _load_npz,
    _movement_summary,
    apply_overlay,
)
from experiments.nested_quantile_base import (
    _competition_delta,
    _issue_bootstrap,
    _ordered_issue_seasons,
)
from src.metrics import CAPACITY_KWH


EXPANSION_ALPHAS = (0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
DEVELOPMENT_COMPONENT_FLOOR = -0.00035


def select_largest_bounded_alpha(
    records: list[dict[str, Any]],
    *,
    component_floor: float = DEVELOPMENT_COMPONENT_FLOOR,
    delta_key: str = "total_delta",
) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record[delta_key]["score"] >= 0.00015
        and record[delta_key]["one_minus_nmae"] >= component_floor
        and record[delta_key]["ficr"] >= component_floor
    ]
    return max(eligible, key=lambda record: record["alpha"]) if eligible else None


def _predictions_for_policy(
    reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    policy: OverlayPolicy,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    candidate = {group: values.copy() for group, values in reference.items()}
    candidate[policy.group], gate = apply_overlay(
        reference[policy.group],
        member[policy.group],
        policy,
    )
    return candidate, gate


def _masked(
    values: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    return {group: prediction[rows] for group, prediction in values.items()}


def _load_selected_policy(path: Path, current_alpha: float) -> OverlayPolicy:
    report = json.loads(path.read_text(encoding="utf-8"))
    raw = report["search"]["exploratory_by_group"]["kpx_group_2"]["policy"]
    source_policy = OverlayPolicy(**raw)
    if source_policy.group != "kpx_group_2" or not np.isclose(
        source_policy.alpha, 0.10
    ):
        raise ValueError("expected the publicly confirmed group-2 gate lineage")
    return OverlayPolicy(**{**asdict(source_policy), "alpha": current_alpha})


def run(
    *,
    kma_oof_path: Path,
    base_run: Path,
    driver_path: Path,
    overlay_report_path: Path,
    gfs_train_path: Path,
    kma_submission_path: Path,
    incumbent_submission_path: Path,
    output_submission_path: Path,
    report_path: Path,
    n_bootstrap: int,
    current_alpha: float = 0.10,
    expansion_alphas: tuple[float, ...] = EXPANSION_ALPHAS,
) -> dict[str, Any]:
    selected_policy = _load_selected_policy(overlay_report_path, current_alpha)
    kma = _load_npz(kma_oof_path)
    nested_path = base_run / "predictions.npz"
    nested = _load_npz(nested_path)
    driver = _load_npz(driver_path)
    kma_index = pd.DatetimeIndex(pd.to_datetime(kma["index_ns"]))
    nested_index = pd.DatetimeIndex(pd.to_datetime(nested["index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver["kpx_group_1__valid_index_ns"])
    )
    common, kma_rows, nested_rows = _align(kma_index, nested_index)
    driver_rows = driver_index.get_indexer(common)
    if (driver_rows < 0).any():
        raise ValueError("driver OOF does not cover aligned KMA/Base v2 rows")

    truth: dict[str, np.ndarray] = {}
    development_reference: dict[str, np.ndarray] = {}
    locked_reference: dict[str, np.ndarray] = {}
    member: dict[str, np.ndarray] = {}
    for group in CAPACITY_KWH:
        truth[group] = nested[f"{group}__truth"][nested_rows].astype(float)
        member[group] = nested[f"{group}__candidate"][nested_rows].astype(float)
        if group == "kpx_group_3":
            development_reference[group] = kma["static_candidate"][
                kma_rows
            ].astype(float)
            locked_reference[group] = kma["rolling_candidate"][
                kma_rows
            ].astype(float)
        else:
            exact = driver[f"{group}__exact_base"][driver_rows].astype(float)
            development_reference[group] = exact
            locked_reference[group] = exact

    development = (common >= Q2_START) & (common < H2_START)
    locked = (common >= H2_START) & (common < LOCKED_END)
    current_development, _ = _predictions_for_policy(
        development_reference,
        member,
        selected_policy,
    )
    development_records: list[dict[str, Any]] = []
    development_variants: dict[float, dict[str, np.ndarray]] = {}
    for alpha in expansion_alphas:
        policy = OverlayPolicy(
            **{**asdict(selected_policy), "alpha": float(alpha)}
        )
        candidate, gate = _predictions_for_policy(
            development_reference,
            member,
            policy,
        )
        development_variants[float(alpha)] = candidate
        development_records.append(
            {
                "alpha": float(alpha),
                "total_delta": _competition_delta(
                    _masked(truth, development),
                    _masked(development_reference, development),
                    _masked(candidate, development),
                )["delta"],
                "incremental_vs_current": _competition_delta(
                    _masked(truth, development),
                    _masked(current_development, development),
                    _masked(candidate, development),
                )["delta"],
                "changed_development_rows": int((gate & development).sum()),
            }
        )
    selection_delta_key = (
        "total_delta"
        if np.isclose(current_alpha, 0.10)
        else "incremental_vs_current"
    )
    selected = select_largest_bounded_alpha(
        development_records,
        delta_key=selection_delta_key,
    )
    if selected is None:
        raise RuntimeError("no bounded group-2 expansion survived Q2")
    expanded_policy = OverlayPolicy(
        **{**asdict(selected_policy), "alpha": selected["alpha"]}
    )

    current_locked, _ = _predictions_for_policy(
        locked_reference,
        member,
        selected_policy,
    )
    expanded_locked, expanded_gate = _predictions_for_policy(
        locked_reference,
        member,
        expanded_policy,
    )
    locked_truth = _masked(truth, locked)
    locked_kma = _masked(locked_reference, locked)
    locked_current = _masked(current_locked, locked)
    locked_expanded = _masked(expanded_locked, locked)
    locked_total = _competition_delta(
        locked_truth,
        locked_kma,
        locked_expanded,
    )
    locked_incremental = _competition_delta(
        locked_truth,
        locked_current,
        locked_expanded,
    )
    locked_index = common[locked]
    issue_times = load_issue_times(gfs_train_path, locked_index)
    seasons, _ = _ordered_issue_seasons(locked_index, issue_times)
    bootstrap_total = _issue_bootstrap(
        locked_truth,
        locked_kma,
        locked_expanded,
        issue_times,
        seasons,
        n_bootstrap=n_bootstrap,
        seed=20_260_725,
    )
    bootstrap_incremental = _issue_bootstrap(
        locked_truth,
        locked_current,
        locked_expanded,
        issue_times,
        seasons,
        n_bootstrap=n_bootstrap,
        seed=20_260_725,
    )
    monthly_total: dict[str, Any] = {}
    monthly_incremental: dict[str, Any] = {}
    periods = locked_index.to_period("M").astype(str)
    for month in dict.fromkeys(periods):
        rows = periods == month
        monthly_total[month] = _competition_delta(
            _masked(locked_truth, rows),
            _masked(locked_kma, rows),
            _masked(locked_expanded, rows),
        )
        monthly_incremental[month] = _competition_delta(
            _masked(locked_truth, rows),
            _masked(locked_current, rows),
            _masked(locked_expanded, rows),
        )

    kma_submission = pd.read_csv(kma_submission_path, encoding="utf-8-sig")
    incumbent_submission = pd.read_csv(
        incumbent_submission_path,
        encoding="utf-8-sig",
    )
    if not kma_submission[["forecast_id", "forecast_kst_dtm"]].equals(
        incumbent_submission[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise ValueError("KMA and alpha-0.10 incumbent submission IDs differ")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(kma_submission["forecast_kst_dtm"])
    )
    nested_test_index = pd.DatetimeIndex(
        pd.to_datetime(nested["test_index_ns"])
    )
    if not test_index.equals(nested_test_index):
        raise ValueError("submission and Base v2 test timestamps differ")
    test_kma = {
        group: kma_submission[group].to_numpy(dtype=float)
        for group in CAPACITY_KWH
    }
    test_member = {
        group: nested[f"{group}__test"].astype(float)
        for group in CAPACITY_KWH
    }
    expected_current, _ = _predictions_for_policy(
        test_kma,
        test_member,
        selected_policy,
    )
    actual_current = {
        group: incumbent_submission[group].to_numpy(dtype=float)
        for group in CAPACITY_KWH
    }
    lineage_max_absolute_error = max(
        float(np.max(np.abs(expected_current[group] - actual_current[group])))
        for group in CAPACITY_KWH
    )
    expanded_test, test_gate = _predictions_for_policy(
        test_kma,
        test_member,
        expanded_policy,
    )
    incremental_movement = _movement_summary(actual_current, expanded_test)
    total_movement = _movement_summary(test_kma, expanded_test)

    minimum_incremental_month = min(
        result["delta"]["score"] for result in monthly_incremental.values()
    )
    positive_incremental_month_fraction = float(
        np.mean(
            [
                result["delta"]["score"] >= 0.0
                for result in monthly_incremental.values()
            ]
        )
    )
    selection_delta = selected[selection_delta_key]
    strict_gates = {
        "incumbent_lineage_exact": lineage_max_absolute_error <= 1e-8,
        "q2_selection_score_minimum": selection_delta["score"] >= 0.00015,
        "q2_selection_components_nonnegative": min(
            selection_delta["one_minus_nmae"],
            selection_delta["ficr"],
        )
        >= 0.0,
        "locked_incremental_score_minimum": locked_incremental["delta"]["score"]
        >= 0.00015,
        "locked_incremental_components_positive": min(
            locked_incremental["delta"]["one_minus_nmae"],
            locked_incremental["delta"]["ficr"],
        )
        > 0.0,
        "all_locked_months_incremental_nonnegative": (
            minimum_incremental_month >= 0.0
        ),
        "incremental_bootstrap_q05_nonnegative": (
            bootstrap_incremental["q05"] >= 0.0
        ),
        "incremental_bootstrap_positive_fraction": (
            bootstrap_incremental["positive_fraction"] >= 0.90
        ),
        "incremental_changed_target_cell_ratio_bounded": (
            incremental_movement["changed_target_cell_ratio"] <= 0.25
        ),
        "incremental_p95_movement_ratio_bounded": (
            incremental_movement["p95_movement_ratio"] <= 0.015
        ),
    }
    exploratory_gates = {
        **{
            key: value
            for key, value in strict_gates.items()
            if key
            not in {
                "q2_selection_components_nonnegative",
                "all_locked_months_incremental_nonnegative",
                "incremental_bootstrap_q05_nonnegative",
                "incremental_bootstrap_positive_fraction",
            }
        },
        "q2_selection_components_tolerable": min(
            selection_delta["one_minus_nmae"],
            selection_delta["ficr"],
        )
        >= DEVELOPMENT_COMPONENT_FLOOR,
        "worst_locked_month_incremental_tolerable": (
            minimum_incremental_month >= -0.001
        ),
        "positive_locked_month_fraction": (
            positive_incremental_month_fraction >= 0.75
        ),
        "incremental_bootstrap_q05_tolerable": (
            bootstrap_incremental["q05"] >= -0.00025
        ),
        "incremental_bootstrap_positive_fraction_tolerable": (
            bootstrap_incremental["positive_fraction"] >= 0.80
        ),
        "total_p95_movement_ratio_bounded": (
            total_movement["p95_movement_ratio"] <= 0.025
        ),
        "total_maximum_movement_ratio_bounded": (
            total_movement["maximum_movement_ratio"] <= 0.06
        ),
    }
    qualified = bool(all(strict_gates.values()))
    decision_tier = (
        "candidate"
        if qualified
        else "exploratory"
        if all(exploratory_gates.values())
        else "rejected"
    )

    candidate_record = None
    if decision_tier != "rejected":
        output = incumbent_submission.copy()
        for group in CAPACITY_KWH:
            output[group] = expanded_test[group]
        output_submission_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_submission_path, index=False, encoding="utf-8-sig")
        candidate_record = {
            "path": output_submission_path.as_posix(),
            "sha256": hashlib.sha256(
                output_submission_path.read_bytes()
            ).hexdigest(),
            "rows": int(len(output)),
            "tier": decision_tier,
            "submission_eligible": qualified,
        }

    report = {
        "method": "public-confirmed group-2 overlay; largest Q2-bounded alpha expansion; H2 locked",
        "sources": {
            "kma_oof": kma_oof_path.as_posix(),
            "base_v2": nested_path.as_posix(),
            "driver_oof": driver_path.as_posix(),
            "selected_overlay_report": overlay_report_path.as_posix(),
            "kma_submission": kma_submission_path.as_posix(),
            "current_public_incumbent": incumbent_submission_path.as_posix(),
        },
        "selection_contract": {
            "current_policy": asdict(selected_policy),
            "candidate_alphas": list(expansion_alphas),
            "selection_delta_key": selection_delta_key,
            "selection": (
                "largest alpha with Q2 selected delta score >= 0.00015 and both "
                f"component deltas >= {DEVELOPMENT_COMPONENT_FLOOR}"
            ),
            "public_score_used_for_selection": False,
        },
        "development": {
            "records": development_records,
            "selected": selected,
            "selected_policy": asdict(expanded_policy),
        },
        "locked_h2": {
            "total_vs_kma": locked_total,
            "incremental_vs_current": locked_incremental,
            "monthly_total_vs_kma": monthly_total,
            "monthly_incremental_vs_current": monthly_incremental,
            "positive_incremental_month_fraction": (
                positive_incremental_month_fraction
            ),
            "changed_rows": int((expanded_gate & locked).sum()),
        },
        "issue_block_bootstrap": {
            "total_vs_kma": bootstrap_total,
            "incremental_vs_current": bootstrap_incremental,
        },
        "test": {
            "lineage_max_absolute_error_kwh": lineage_max_absolute_error,
            "changed_rows": int(test_gate.sum()),
            "incremental_movement": incremental_movement,
            "total_movement_vs_kma": total_movement,
        },
        "promotion_gates": strict_gates,
        "exploratory_gates": exploratory_gates,
        "decision_tier": decision_tier,
        "candidate": candidate_record,
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
        "--kma-oof",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof_20260725.npz",
    )
    parser.add_argument(
        "--base-run",
        default="artifacts_final/base_v2/group3_curtailment_full_20260725",
    )
    parser.add_argument(
        "--driver-cache",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    parser.add_argument(
        "--overlay-report",
        default="artifacts_final/base_v2/kma_incumbent_local_overlay_20260725.json",
    )
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--kma-submission",
        default="submissions/blend_best_kma_um_power_curve_gate.csv",
    )
    parser.add_argument(
        "--incumbent-submission",
        default="submissions/archive/kma_incumbent_basev2_local_overlay_20260725.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="artifacts_final/candidates/kma_group2_overlay_alpha20_20260725.csv",
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/base_v2/kma_group2_overlay_expansion_20260725.json",
    )
    parser.add_argument("--current-alpha", type=float, default=0.10)
    parser.add_argument(
        "--candidate-alphas",
        default=",".join(str(value) for value in EXPANSION_ALPHAS),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    expansion_alphas = tuple(
        float(value.strip())
        for value in args.candidate_alphas.split(",")
        if value.strip()
    )
    if (
        not expansion_alphas
        or any(alpha <= args.current_alpha for alpha in expansion_alphas)
        or tuple(sorted(expansion_alphas)) != expansion_alphas
    ):
        raise ValueError(
            "candidate alphas must be sorted and greater than current alpha"
        )
    report = run(
        kma_oof_path=Path(args.kma_oof),
        base_run=Path(args.base_run),
        driver_path=Path(args.driver_cache),
        overlay_report_path=Path(args.overlay_report),
        gfs_train_path=Path(args.gfs_train),
        kma_submission_path=Path(args.kma_submission),
        incumbent_submission_path=Path(args.incumbent_submission),
        output_submission_path=Path(args.output_submission),
        report_path=Path(args.report),
        n_bootstrap=args.n_bootstrap,
        current_alpha=args.current_alpha,
        expansion_alphas=expansion_alphas,
    )
    print(
        json.dumps(
            {
                "selected": report["development"]["selected"],
                "locked_incremental": report["locked_h2"][
                    "incremental_vs_current"
                ],
                "bootstrap_incremental": report["issue_block_bootstrap"][
                    "incremental_vs_current"
                ],
                "test": report["test"],
                "decision_tier": report["decision_tier"],
                "candidate": report["candidate"],
                "report": args.report,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
