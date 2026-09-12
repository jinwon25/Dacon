from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.nested_quantile_base import (
    _competition_delta,
    _issue_bootstrap,
    _ordered_issue_seasons,
)
from src.metrics import CAPACITY_KWH, evaluate_group


Q2_START = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
LOCKED_END = pd.Timestamp("2024-12-01")


@dataclass(frozen=True)
class OverlayPolicy:
    group: str
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_disagreement_kwh: float
    alpha: float


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def _align(
    left: pd.DatetimeIndex,
    right: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    common = left.intersection(right)
    if common.empty:
        raise ValueError("KMA and Base v2 OOF caches have no common timestamps")
    return common, left.get_indexer(common), right.get_indexer(common)


def apply_overlay(
    reference: np.ndarray,
    member: np.ndarray,
    policy: OverlayPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    capacity = CAPACITY_KWH[policy.group]
    disagreement = member - reference
    if policy.direction == "up":
        direction = disagreement > 0.0
    elif policy.direction == "down":
        direction = disagreement < 0.0
    elif policy.direction == "both":
        direction = np.ones(len(reference), dtype=bool)
    else:
        raise ValueError(f"unknown overlay direction: {policy.direction}")
    ratio = reference / capacity
    gate = (
        direction
        & (np.abs(disagreement) >= policy.minimum_disagreement_kwh)
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )
    candidate = reference.copy()
    candidate[gate] = np.clip(
        reference[gate] + policy.alpha * disagreement[gate],
        0.0,
        capacity,
    )
    return candidate, gate


def _group_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
    capacity: float,
) -> dict[str, float]:
    before = evaluate_group(truth[rows], reference[rows], capacity)
    after = evaluate_group(truth[rows], candidate[rows], capacity)
    return {
        "score": float((after.score - before.score) / 3.0),
        "one_minus_nmae": float(
            (after.one_minus_nmae - before.one_minus_nmae) / 3.0
        ),
        "ficr": float((after.ficr - before.ficr) / 3.0),
    }


def select_development_policies(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    development: np.ndarray,
) -> dict[str, dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    for group, capacity in CAPACITY_KWH.items():
        disagreement = member[group] - reference[group]
        for direction in ("both", "up", "down"):
            if direction == "up":
                pool_mask = development & (disagreement > 0.0)
            elif direction == "down":
                pool_mask = development & (disagreement < 0.0)
            else:
                pool_mask = development.copy()
            pool = np.abs(disagreement[pool_mask])
            if not len(pool):
                continue
            for coverage in (0.75, 0.50, 0.25, 0.10):
                threshold = float(np.quantile(pool, 1.0 - coverage))
                for minimum_ratio, maximum_ratio in (
                    (0.10, 1.00),
                    (0.10, 0.80),
                    (0.20, 0.80),
                    (0.20, 0.60),
                ):
                    for alpha in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
                        policy = OverlayPolicy(
                            group=group,
                            direction=direction,
                            coverage=coverage,
                            minimum_base_ratio=minimum_ratio,
                            maximum_base_ratio=maximum_ratio,
                            minimum_disagreement_kwh=threshold,
                            alpha=alpha,
                        )
                        candidate, gate = apply_overlay(
                            reference[group], member[group], policy
                        )
                        changed = int((gate & development).sum())
                        if not changed:
                            continue
                        delta = _group_delta(
                            truth[group],
                            reference[group],
                            candidate,
                            development,
                            capacity,
                        )
                        rows.append(
                            {
                                "policy": policy,
                                "delta": delta,
                                "changed_development_rows": changed,
                                "development_rows": int(development.sum()),
                            }
                        )

    def choose(
        component_floor: float,
        group: str | None = None,
    ) -> dict[str, Any] | None:
        eligible = [
            row
            for row in rows
            if (group is None or row["policy"].group == group)
            if row["delta"]["score"] >= 0.00015
            and row["delta"]["one_minus_nmae"] >= component_floor
            and row["delta"]["ficr"] >= component_floor
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda row: (
                row["delta"]["score"],
                min(
                    row["delta"]["one_minus_nmae"],
                    row["delta"]["ficr"],
                ),
                -row["changed_development_rows"],
                -row["policy"].alpha,
            ),
        )

    return {
        "strict": choose(0.0),
        "exploratory": choose(-0.00035),
        "strict_by_group": {
            group: choose(0.0, group) for group in CAPACITY_KWH
        },
        "exploratory_by_group": {
            group: choose(-0.00035, group) for group in CAPACITY_KWH
        },
        "searched": {
            "policies": len(rows),
            "strict_eligible": sum(
                row["delta"]["score"] >= 0.00015
                and row["delta"]["one_minus_nmae"] >= 0.0
                and row["delta"]["ficr"] >= 0.0
                for row in rows
            ),
            "exploratory_eligible": sum(
                row["delta"]["score"] >= 0.00015
                and row["delta"]["one_minus_nmae"] >= -0.00035
                and row["delta"]["ficr"] >= -0.00035
                for row in rows
            ),
        },
    }


def _serialize_selection(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    if selection is None:
        return None
    return {
        "policy": asdict(selection["policy"]),
        "delta": selection["delta"],
        "changed_development_rows": selection["changed_development_rows"],
        "development_rows": selection["development_rows"],
    }


def _movement_summary(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    normalized: list[np.ndarray] = []
    changed = 0
    total = 0
    by_group: dict[str, Any] = {}
    for group, capacity in CAPACITY_KWH.items():
        absolute = np.abs(candidate[group] - reference[group])
        local_changed = absolute > 1e-6
        normalized.append(absolute / capacity)
        changed += int(local_changed.sum())
        total += len(absolute)
        by_group[group] = {
            "changed_rows": int(local_changed.sum()),
            "changed_ratio": float(local_changed.mean()),
            "mean_absolute_kwh": float(absolute.mean()),
            "p95_absolute_kwh": float(np.quantile(absolute, 0.95)),
            "maximum_absolute_kwh": float(absolute.max()),
        }
    normalized_vector = np.concatenate(normalized)
    return {
        "changed_target_cells": changed,
        "total_target_cells": total,
        "changed_target_cell_ratio": float(changed / total),
        "p95_movement_ratio": float(np.quantile(normalized_vector, 0.95)),
        "maximum_movement_ratio": float(normalized_vector.max()),
        "by_group": by_group,
    }


def evaluate_selected_policy(
    selection: dict[str, Any],
    *,
    common: pd.DatetimeIndex,
    truth: dict[str, np.ndarray],
    locked_reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    locked: np.ndarray,
    issue_times: pd.DatetimeIndex,
    test_reference: dict[str, np.ndarray],
    test_member: dict[str, np.ndarray],
    n_bootstrap: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = selection["policy"]
    locked_candidate = {
        group: values.copy() for group, values in locked_reference.items()
    }
    locked_candidate[policy.group], locked_gate = apply_overlay(
        locked_reference[policy.group],
        member[policy.group],
        policy,
    )
    locked_truth = {group: values[locked] for group, values in truth.items()}
    locked_before = {
        group: values[locked] for group, values in locked_reference.items()
    }
    locked_after = {
        group: values[locked] for group, values in locked_candidate.items()
    }
    locked_result = _competition_delta(
        locked_truth,
        locked_before,
        locked_after,
    )
    locked_index = common[locked]
    seasons, _ = _ordered_issue_seasons(locked_index, issue_times)
    bootstrap = _issue_bootstrap(
        locked_truth,
        locked_before,
        locked_after,
        issue_times,
        seasons,
        n_bootstrap=n_bootstrap,
        seed=20_260_725,
    )
    monthly: dict[str, Any] = {}
    locked_periods = locked_index.to_period("M").astype(str)
    for month in dict.fromkeys(locked_periods):
        rows = locked_periods == month
        monthly[month] = _competition_delta(
            {group: values[rows] for group, values in locked_truth.items()},
            {group: values[rows] for group, values in locked_before.items()},
            {group: values[rows] for group, values in locked_after.items()},
        )

    test_candidate = {
        group: values.copy() for group, values in test_reference.items()
    }
    test_candidate[policy.group], test_gate = apply_overlay(
        test_reference[policy.group],
        test_member[policy.group],
        policy,
    )
    movement = _movement_summary(test_reference, test_candidate)
    minimum_month = min(
        result["delta"]["score"] for result in monthly.values()
    )
    positive_month_fraction = float(
        np.mean(
            [
                result["delta"]["score"] >= 0.0
                for result in monthly.values()
            ]
        )
    )
    strict_gates = {
        "development_score_minimum": selection["delta"]["score"] >= 0.00015,
        "development_components_nonnegative": min(
            selection["delta"]["one_minus_nmae"],
            selection["delta"]["ficr"],
        )
        >= 0.0,
        "locked_score_minimum": locked_result["delta"]["score"] >= 0.00015,
        "locked_one_minus_nmae_nonnegative": locked_result["delta"][
            "one_minus_nmae"
        ]
        >= 0.0,
        "locked_ficr_nonnegative": locked_result["delta"]["ficr"] >= 0.0,
        "worst_locked_month_nonnegative": minimum_month >= 0.0,
        "bootstrap_q05_nonnegative": bootstrap["q05"] >= 0.0,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.90,
        "changed_target_cell_ratio_bounded": (
            movement["changed_target_cell_ratio"] <= 0.25
        ),
        "p95_movement_ratio_bounded": movement["p95_movement_ratio"] <= 0.015,
    }
    exploratory_gates = {
        "development_score_minimum": selection["delta"]["score"] >= 0.00015,
        "development_components_tolerable": min(
            selection["delta"]["one_minus_nmae"],
            selection["delta"]["ficr"],
        )
        >= -0.00035,
        "locked_score_minimum": locked_result["delta"]["score"] >= 0.00015,
        "locked_one_minus_nmae_tolerable": locked_result["delta"][
            "one_minus_nmae"
        ]
        >= -0.00035,
        "locked_ficr_tolerable": locked_result["delta"]["ficr"] >= -0.00035,
        "worst_locked_month_tolerable": minimum_month >= -0.001,
        "positive_locked_month_fraction": positive_month_fraction >= 0.75,
        "bootstrap_q05_tolerable": bootstrap["q05"] >= -0.00025,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.80,
        "changed_target_cell_ratio_bounded": (
            movement["changed_target_cell_ratio"] <= 0.50
        ),
        "p95_movement_ratio_bounded": movement["p95_movement_ratio"] <= 0.025,
    }
    tier = (
        "candidate"
        if all(strict_gates.values())
        else "exploratory"
        if all(exploratory_gates.values())
        else "rejected"
    )
    return (
        {
            "selection": _serialize_selection(selection),
            "locked_h2": locked_result,
            "locked_monthly": monthly,
            "positive_locked_month_fraction": positive_month_fraction,
            "issue_block_bootstrap": bootstrap,
            "locked_changed_rows": int((locked_gate & locked).sum()),
            "test_changed_rows": int(test_gate.sum()),
            "test_movement": movement,
            "promotion_gates": strict_gates,
            "exploratory_gates": exploratory_gates,
            "decision_tier": tier,
        },
        test_candidate,
    )


def run(
    *,
    kma_oof_path: Path,
    base_run: Path,
    driver_path: Path,
    gfs_train_path: Path,
    kma_submission_path: Path,
    output_submission_path: Path,
    report_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    kma = _load_npz(kma_oof_path)
    nested = _load_npz(base_run / "predictions.npz")
    driver = _load_npz(driver_path)
    kma_index = pd.DatetimeIndex(pd.to_datetime(kma["index_ns"]))
    nested_index = pd.DatetimeIndex(pd.to_datetime(nested["index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver["kpx_group_1__valid_index_ns"])
    )
    common, kma_rows, nested_rows = _align(kma_index, nested_index)
    driver_rows = driver_index.get_indexer(common)
    if (driver_rows < 0).any():
        raise ValueError("driver OOF does not cover the KMA/Base v2 intersection")

    truth: dict[str, np.ndarray] = {}
    development_reference: dict[str, np.ndarray] = {}
    locked_reference: dict[str, np.ndarray] = {}
    member: dict[str, np.ndarray] = {}
    for group in CAPACITY_KWH:
        nested_truth = nested[f"{group}__truth"][nested_rows].astype(float)
        driver_truth = driver[f"{group}__valid_truth"][driver_rows].astype(float)
        if not np.allclose(nested_truth, driver_truth, atol=1e-5, rtol=0.0):
            raise ValueError(f"truth mismatch for {group}")
        truth[group] = nested_truth
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
    if not development.any() or not locked.any():
        raise ValueError("aligned OOF does not cover development and locked periods")
    selections = select_development_policies(
        truth,
        development_reference,
        member,
        development,
    )

    kma_submission = pd.read_csv(kma_submission_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(kma_submission["forecast_kst_dtm"])
    )
    nested_test_index = pd.DatetimeIndex(
        pd.to_datetime(nested["test_index_ns"])
    )
    if not test_index.equals(nested_test_index):
        raise ValueError("KMA submission and Base v2 test timestamps differ")
    test_reference = {
        group: kma_submission[group].to_numpy(dtype=float)
        for group in CAPACITY_KWH
    }
    test_member = {
        group: nested[f"{group}__test"].astype(float)
        for group in CAPACITY_KWH
    }
    issue_times = load_issue_times(gfs_train_path, common[locked])

    variants: dict[str, Any] = {}
    candidate_arrays: dict[str, dict[str, np.ndarray]] = {}
    seen: set[OverlayPolicy] = set()
    contracts: list[tuple[str, dict[str, Any] | None]] = []
    for tier in ("strict", "exploratory"):
        for group in CAPACITY_KWH:
            contracts.append(
                (f"{tier}_{group}", selections[f"{tier}_by_group"][group])
            )
    for name, selection in contracts:
        if selection is None or selection["policy"] in seen:
            continue
        seen.add(selection["policy"])
        result, arrays = evaluate_selected_policy(
            selection,
            common=common,
            truth=truth,
            locked_reference=locked_reference,
            member=member,
            locked=locked,
            issue_times=issue_times,
            test_reference=test_reference,
            test_member=test_member,
            n_bootstrap=n_bootstrap,
        )
        variants[name] = result
        candidate_arrays[name] = arrays

    viable = [
        name
        for name, result in variants.items()
        if result["decision_tier"] in {"candidate", "exploratory"}
    ]
    selected_name = (
        max(
            viable,
            key=lambda name: (
                variants[name]["decision_tier"] == "candidate",
                variants[name]["selection"]["delta"]["score"],
                -variants[name]["test_movement"]["changed_target_cell_ratio"],
            ),
        )
        if viable
        else (
            max(
                variants,
                key=lambda name: variants[name]["selection"]["delta"]["score"],
            )
            if variants
            else None
        )
    )
    selected_tier = (
        variants[selected_name]["decision_tier"]
        if selected_name is not None
        else "rejected"
    )
    candidate_record = None
    if selected_name is not None and selected_tier != "rejected":
        output = kma_submission.copy()
        for group in CAPACITY_KWH:
            output[group] = candidate_arrays[selected_name][group]
        output_submission_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_submission_path, index=False, encoding="utf-8-sig")
        candidate_record = {
            "path": output_submission_path.as_posix(),
            "sha256": hashlib.sha256(
                output_submission_path.read_bytes()
            ).hexdigest(),
            "rows": int(len(output)),
            "tier": selected_tier,
            "submission_eligible": selected_tier == "candidate",
        }

    report = {
        "method": "Q2-selected single-group bounded Base v2 overlay on KMA incumbent; H2 locked",
        "sources": {
            "kma_oof": kma_oof_path.as_posix(),
            "base_v2": (base_run / "predictions.npz").as_posix(),
            "driver_oof": driver_path.as_posix(),
            "kma_public_incumbent": kma_submission_path.as_posix(),
        },
        "periods": {
            "development": "2024-Q2",
            "development_rows": int(development.sum()),
            "locked": "2024-07-01 01:00 through 2024-11-30 23:00",
            "locked_rows": int(locked.sum()),
        },
        "search": {
            **selections["searched"],
            "strict_selection": _serialize_selection(selections["strict"]),
            "exploratory_selection": _serialize_selection(
                selections["exploratory"]
            ),
            "strict_by_group": {
                group: _serialize_selection(selection)
                for group, selection in selections["strict_by_group"].items()
            },
            "exploratory_by_group": {
                group: _serialize_selection(selection)
                for group, selection in selections[
                    "exploratory_by_group"
                ].items()
            },
        },
        "variants": variants,
        "selected_variant": selected_name,
        "decision_tier": selected_tier,
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
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--kma-submission",
        default="submissions/blend_best_kma_um_power_curve_gate.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="artifacts_final/candidates/kma_incumbent_basev2_local_overlay_20260725.csv",
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/base_v2/kma_incumbent_local_overlay_20260725.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run(
        kma_oof_path=Path(args.kma_oof),
        base_run=Path(args.base_run),
        driver_path=Path(args.driver_cache),
        gfs_train_path=Path(args.gfs_train),
        kma_submission_path=Path(args.kma_submission),
        output_submission_path=Path(args.output_submission),
        report_path=Path(args.report),
        n_bootstrap=args.n_bootstrap,
    )
    print(
        json.dumps(
            {
                "search": report["search"],
                "selected_variant": report["selected_variant"],
                "decision_tier": report["decision_tier"],
                "selected_result": (
                    report["variants"].get(report["selected_variant"])
                    if report["selected_variant"] is not None
                    else None
                ),
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
