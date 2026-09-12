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


GROUP = "kpx_group_2"
CORE_ALPHA = 0.2375


@dataclass(frozen=True)
class SupplementPolicy:
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_disagreement_kwh: float
    alpha: float


def _masked(
    values: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    return {group: prediction[rows] for group, prediction in values.items()}


def _apply_core(
    reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    core_policy: OverlayPolicy,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    candidate = {group: values.copy() for group, values in reference.items()}
    candidate[GROUP], gate = apply_overlay(
        reference[GROUP],
        member[GROUP],
        core_policy,
    )
    return candidate, gate


def apply_supplement(
    core_candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    core_gate: np.ndarray,
    policy: SupplementPolicy,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    disagreement = member[GROUP] - reference[GROUP]
    if policy.direction == "up":
        direction = disagreement > 0.0
    elif policy.direction == "down":
        direction = disagreement < 0.0
    elif policy.direction == "both":
        direction = np.ones(len(disagreement), dtype=bool)
    else:
        raise ValueError(f"unknown direction: {policy.direction}")
    ratio = reference[GROUP] / CAPACITY_KWH[GROUP]
    gate = (
        ~core_gate
        & direction
        & (np.abs(disagreement) >= policy.minimum_disagreement_kwh)
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )
    candidate = {
        group: values.copy() for group, values in core_candidate.items()
    }
    candidate[GROUP][gate] = np.clip(
        core_candidate[GROUP][gate] + policy.alpha * disagreement[gate],
        0.0,
        CAPACITY_KWH[GROUP],
    )
    return candidate, gate


def select_supplement(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    member: dict[str, np.ndarray],
    core_candidate: dict[str, np.ndarray],
    core_gate: np.ndarray,
    development: np.ndarray,
) -> tuple[dict[str, Any] | None, int]:
    disagreement = member[GROUP] - reference[GROUP]
    rows: list[dict[str, Any]] = []
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = disagreement > 0.0
        elif direction_name == "down":
            direction = disagreement < 0.0
        else:
            direction = np.ones(len(disagreement), dtype=bool)
        pool = development & ~core_gate & direction
        magnitudes = np.abs(disagreement[pool])
        if not len(magnitudes):
            continue
        for coverage in (0.05, 0.10, 0.25, 0.50, 1.00):
            threshold = (
                0.0
                if coverage == 1.00
                else float(np.quantile(magnitudes, 1.0 - coverage))
            )
            for minimum_ratio, maximum_ratio in (
                (0.10, 1.00),
                (0.10, 0.80),
                (0.20, 0.80),
                (0.20, 0.60),
                (0.40, 1.00),
            ):
                for alpha in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
                    policy = SupplementPolicy(
                        direction=direction_name,
                        coverage=coverage,
                        minimum_base_ratio=minimum_ratio,
                        maximum_base_ratio=maximum_ratio,
                        minimum_disagreement_kwh=threshold,
                        alpha=alpha,
                    )
                    candidate, gate = apply_supplement(
                        core_candidate,
                        reference,
                        member,
                        core_gate,
                        policy,
                    )
                    changed = int((gate & development).sum())
                    if not changed:
                        continue
                    delta = _competition_delta(
                        _masked(truth, development),
                        _masked(core_candidate, development),
                        _masked(candidate, development),
                    )["delta"]
                    rows.append(
                        {
                            "policy": policy,
                            "delta": delta,
                            "changed_development_rows": changed,
                        }
                    )
    eligible = [
        row
        for row in rows
        if row["delta"]["score"] >= 0.00005
        and row["delta"]["one_minus_nmae"] >= -0.00035
        and row["delta"]["ficr"] >= -0.00035
    ]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["delta"]["score"],
                min(
                    row["delta"]["one_minus_nmae"],
                    row["delta"]["ficr"],
                ),
                -row["changed_development_rows"],
            ),
        )
        if eligible
        else None
    )
    return selected, len(rows)


def _load_core_policy(path: Path) -> OverlayPolicy:
    report = json.loads(path.read_text(encoding="utf-8"))
    raw = report["search"]["exploratory_by_group"][GROUP]["policy"]
    source = OverlayPolicy(**raw)
    return OverlayPolicy(**{**asdict(source), "alpha": CORE_ALPHA})


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
) -> dict[str, Any]:
    core_policy = _load_core_policy(overlay_report_path)
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
        raise ValueError("driver OOF does not cover aligned rows")

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
    development_core, development_core_gate = _apply_core(
        development_reference,
        member,
        core_policy,
    )
    selected, searched = select_supplement(
        truth,
        development_reference,
        member,
        development_core,
        development_core_gate,
        development,
    )
    if selected is None:
        raise RuntimeError("no Q2 supplement survived")
    policy = selected["policy"]

    locked_core, locked_core_gate = _apply_core(
        locked_reference,
        member,
        core_policy,
    )
    locked_candidate, locked_supplement_gate = apply_supplement(
        locked_core,
        locked_reference,
        member,
        locked_core_gate,
        policy,
    )
    locked_truth = _masked(truth, locked)
    locked_before = _masked(locked_core, locked)
    locked_after = _masked(locked_candidate, locked)
    locked_result = _competition_delta(
        locked_truth,
        locked_before,
        locked_after,
    )
    locked_index = common[locked]
    issue_times = load_issue_times(gfs_train_path, locked_index)
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
    periods = locked_index.to_period("M").astype(str)
    for month in dict.fromkeys(periods):
        rows = periods == month
        monthly[month] = _competition_delta(
            _masked(locked_truth, rows),
            _masked(locked_before, rows),
            _masked(locked_after, rows),
        )

    kma_submission = pd.read_csv(kma_submission_path, encoding="utf-8-sig")
    incumbent = pd.read_csv(incumbent_submission_path, encoding="utf-8-sig")
    if not kma_submission[["forecast_id", "forecast_kst_dtm"]].equals(
        incumbent[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise ValueError("KMA and incumbent IDs differ")
    test_kma = {
        group: kma_submission[group].to_numpy(dtype=float)
        for group in CAPACITY_KWH
    }
    test_member = {
        group: nested[f"{group}__test"].astype(float)
        for group in CAPACITY_KWH
    }
    expected_core, test_core_gate = _apply_core(
        test_kma,
        test_member,
        core_policy,
    )
    actual_core = {
        group: incumbent[group].to_numpy(dtype=float)
        for group in CAPACITY_KWH
    }
    lineage_error = max(
        float(np.max(np.abs(expected_core[group] - actual_core[group])))
        for group in CAPACITY_KWH
    )
    test_candidate, test_supplement_gate = apply_supplement(
        actual_core,
        test_kma,
        test_member,
        test_core_gate,
        policy,
    )
    incremental_movement = _movement_summary(actual_core, test_candidate)
    total_movement = _movement_summary(test_kma, test_candidate)
    monthly_scores = [
        result["delta"]["score"] for result in monthly.values()
    ]
    exploratory_gates = {
        "incumbent_lineage_exact": lineage_error <= 1e-8,
        "q2_score_minimum": selected["delta"]["score"] >= 0.00005,
        "q2_components_tolerable": min(
            selected["delta"]["one_minus_nmae"],
            selected["delta"]["ficr"],
        )
        >= -0.00035,
        "locked_score_minimum": locked_result["delta"]["score"] >= 0.00015,
        "locked_components_tolerable": min(
            locked_result["delta"]["one_minus_nmae"],
            locked_result["delta"]["ficr"],
        )
        >= -0.00035,
        "worst_locked_month_tolerable": min(monthly_scores) >= -0.001,
        "positive_locked_month_fraction": (
            float(np.mean(np.asarray(monthly_scores) >= 0.0)) >= 0.75
        ),
        "bootstrap_q05_tolerable": bootstrap["q05"] >= -0.00025,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.80,
        "total_changed_target_cell_ratio_bounded": (
            total_movement["changed_target_cell_ratio"] <= 0.25
        ),
        "total_p95_movement_ratio_bounded": (
            total_movement["p95_movement_ratio"] <= 0.025
        ),
        "total_maximum_movement_ratio_bounded": (
            total_movement["maximum_movement_ratio"] <= 0.06
        ),
    }
    decision_tier = (
        "exploratory" if all(exploratory_gates.values()) else "rejected"
    )
    candidate_record = None
    if decision_tier == "exploratory":
        output = incumbent.copy()
        for group in CAPACITY_KWH:
            output[group] = test_candidate[group]
        output_submission_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_submission_path, index=False, encoding="utf-8-sig")
        candidate_record = {
            "path": output_submission_path.as_posix(),
            "sha256": hashlib.sha256(
                output_submission_path.read_bytes()
            ).hexdigest(),
            "rows": int(len(output)),
            "tier": decision_tier,
            "submission_eligible": False,
        }

    report = {
        "method": "alpha23.75 core fixed; Q2-selected non-core sparse group-2 supplement; H2 locked",
        "sources": {
            "kma_oof": kma_oof_path.as_posix(),
            "base_v2": nested_path.as_posix(),
            "driver_oof": driver_path.as_posix(),
            "overlay_report": overlay_report_path.as_posix(),
            "public_incumbent": incumbent_submission_path.as_posix(),
        },
        "selection_contract": {
            "core_policy": asdict(core_policy),
            "searched_policies": searched,
            "public_score_used_for_selection": False,
        },
        "development": {
            "selected_policy": asdict(policy),
            "delta": selected["delta"],
            "changed_rows": selected["changed_development_rows"],
        },
        "locked_h2": {
            "incremental": locked_result,
            "monthly_incremental": monthly,
            "changed_rows": int((locked_supplement_gate & locked).sum()),
        },
        "issue_block_bootstrap": bootstrap,
        "test": {
            "lineage_max_absolute_error_kwh": lineage_error,
            "supplement_changed_rows": int(test_supplement_gate.sum()),
            "incremental_movement": incremental_movement,
            "total_movement_vs_kma": total_movement,
        },
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
        default="artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="artifacts_final/candidates/kma_group2_alpha2375_up_supplement_20260725.csv",
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/base_v2/kma_group2_alpha2375_up_supplement_20260725.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
