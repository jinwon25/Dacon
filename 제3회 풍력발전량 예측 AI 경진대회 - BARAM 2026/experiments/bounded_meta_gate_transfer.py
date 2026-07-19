from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.exact_oof_meta_gate import (
    H2_START,
    _bootstrap_days,
    _evaluate_period,
    apply_meta_gate,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from src.metrics import CAPACITY_KWH


TARGET = "kpx_group_3"
MACRO_GROUP_COUNT = 3
REFERENCE_POLICY = {"threshold": 0.55, "alpha": 0.25}
FINE_POLICY = {"threshold": 0.545, "alpha": 0.50}
BOUNDED_POLICY = {"threshold": 0.55, "alpha": 0.50}


def component_deltas(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - reference[key])
        for key in ("score", "one_minus_nmae", "ficr")
    }


def macro_transfer_ratios(
    public_delta: dict[str, float], locked_group3_delta: dict[str, float]
) -> dict[str, float | None]:
    """Compare public macro deltas to the matching one-third OOF group delta."""
    ratios: dict[str, float | None] = {}
    for key, public_value in public_delta.items():
        local_macro_value = float(locked_group3_delta[key]) / MACRO_GROUP_COUNT
        ratios[key] = (
            float(public_value) / local_macro_value
            if abs(local_macro_value) > 1e-15
            else None
        )
    return ratios


def build_strong_gate_candidate(
    pre_meta: pd.DataFrame, reference_meta: pd.DataFrame
) -> pd.DataFrame:
    """Double only the already deployed p>=.55 alpha=.25 movement to alpha=.50."""
    if pre_meta.columns.tolist() != reference_meta.columns.tolist():
        raise ValueError("Source submission schemas do not match")
    if len(pre_meta) != len(reference_meta):
        raise ValueError("Source submission row counts do not match")
    for identity in ("forecast_id", "forecast_kst_dtm"):
        if not pre_meta[identity].equals(reference_meta[identity]):
            raise ValueError(f"Source submission {identity} values do not match")
    output = reference_meta.copy()
    current = pre_meta[TARGET].to_numpy(dtype=float)
    alpha25 = reference_meta[TARGET].to_numpy(dtype=float)
    output[TARGET] = np.clip(
        current + 2.0 * (alpha25 - current),
        0.0,
        CAPACITY_KWH[TARGET],
    )
    return output


def run(
    labels_path: Path,
    driver_path: Path,
    meta_cache_path: Path,
    pre_meta_path: Path,
    reference_meta_path: Path,
    fine_path: Path,
    meta_sweep_report_path: Path,
    settlement_report_path: Path,
    output_path: Path,
    report_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    sweep_report = json.loads(meta_sweep_report_path.read_text(encoding="utf-8"))
    settlement_report = json.loads(settlement_report_path.read_text(encoding="utf-8"))

    public_reference = {
        "score": 0.6416553726,
        "one_minus_nmae": 0.8754552834,
        "ficr": 0.4078554618,
    }
    public_fine = {
        "score": 0.6417471627,
        "one_minus_nmae": 0.8754733572,
        "ficr": 0.4080209682,
    }
    public_settlement = {
        "score": 0.6377660509,
        "one_minus_nmae": 0.8714368889,
        "ficr": 0.4040952129,
    }
    public_fine_delta = component_deltas(public_fine, public_reference)
    locked_fine_group3_delta = sweep_report["locked_incremental_over_reference"][
        "metrics"
    ]["delta"]
    fine_transfer = macro_transfer_ratios(
        public_fine_delta, locked_fine_group3_delta
    )

    public_settlement_delta = component_deltas(public_settlement, public_reference)
    locked_settlement_macro_delta = settlement_report["locked_h2_macro_delta"]
    settlement_transfer = {
        key: public_settlement_delta[key] / float(locked_settlement_macro_delta[key])
        for key in public_settlement_delta
    }

    (
        _labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        _features,
        action,
    ) = _prepare_validation(labels_path, driver_path)
    cache = np.load(meta_cache_path)
    cache_index = pd.DatetimeIndex(pd.to_datetime(cache["valid_index_ns"]))
    if not index.equals(cache_index):
        raise ValueError("Meta cache and exact OOF validation indices do not match")
    probability = cache["valid_probability"].astype(float)
    h2 = index >= H2_START

    reference_oof, _ = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=REFERENCE_POLICY["threshold"],
        extra_alpha=REFERENCE_POLICY["alpha"],
    )
    fine_oof, fine_gate = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=FINE_POLICY["threshold"],
        extra_alpha=FINE_POLICY["alpha"],
    )
    bounded_oof, bounded_gate = apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=BOUNDED_POLICY["threshold"],
        extra_alpha=BOUNDED_POLICY["alpha"],
    )
    bounded_vs_reference = _evaluate_period(
        truth, reference_oof, bounded_oof, h2
    )["delta"]
    bounded_vs_fine = _evaluate_period(truth, fine_oof, bounded_oof, h2)["delta"]
    monthly_vs_reference: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        month_mask = h2 & (index.month == month)
        monthly_vs_reference[str(month)] = _evaluate_period(
            truth, reference_oof, bounded_oof, month_mask
        )["delta"]
    bootstrap = _bootstrap_days(
        truth,
        reference_oof,
        bounded_oof,
        index,
        h2,
        n_bootstrap,
        seed=20260722,
    )

    pre_meta = pd.read_csv(pre_meta_path, encoding="utf-8-sig")
    reference_meta = pd.read_csv(reference_meta_path, encoding="utf-8-sig")
    fine = pd.read_csv(fine_path, encoding="utf-8-sig")
    output = build_strong_gate_candidate(pre_meta, reference_meta)
    if not output[[*CAPACITY_KWH]].equals(reference_meta[[*CAPACITY_KWH]].assign(
        **{TARGET: output[TARGET]}
    )):
        raise ValueError("The bounded candidate unexpectedly changed groups 1 or 2")
    source_target = reference_meta[TARGET].to_numpy(dtype=float)
    candidate_target = output[TARGET].to_numpy(dtype=float)
    fine_target = fine[TARGET].to_numpy(dtype=float)
    movement = np.abs(candidate_target - source_target)
    difference_from_fine = np.abs(candidate_target - fine_target)
    locked_macro_gain_vs_fine = float(bounded_vs_fine["score"]) / MACRO_GROUP_COUNT
    conservative_transfer = max(0.0, min(1.0, float(fine_transfer["score"] or 0.0)))
    projected_public_gain_vs_fine = locked_macro_gain_vs_fine * conservative_transfer

    # Saturation guard: a candidate this close to the current public best should be
    # retained for lineage, but must not consume a submission unless its projected
    # gain is material and the locked distribution is convincingly positive.
    promote = bool(
        locked_macro_gain_vs_fine >= 0.00005
        and projected_public_gain_vs_fine >= 0.00002
        and bounded_vs_fine["one_minus_nmae"] >= 0.0
        and bounded_vs_fine["ficr"] >= 0.0
        and bootstrap["positive_fraction"] >= 0.90
        and bootstrap["q05"] >= 0.0
    )
    if promote:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "method": "public-transfer audit plus bounded strong-meta-gate decomposition",
        "public_feedback": {
            "reference_submission_id": 1494307,
            "fine_submission_id": 1494670,
            "settlement_submission_id": 1494668,
            "fine_delta": public_fine_delta,
            "fine_locked_group3_delta": locked_fine_group3_delta,
            "fine_locked_macro_delta": {
                key: float(value) / MACRO_GROUP_COUNT
                for key, value in locked_fine_group3_delta.items()
            },
            "fine_public_transfer_ratio": fine_transfer,
            "settlement_delta": public_settlement_delta,
            "settlement_locked_macro_delta": locked_settlement_macro_delta,
            "settlement_public_transfer_ratio": settlement_transfer,
            "conclusion": (
                "The fine meta-gate direction transferred weakly but positively. "
                "Broad settlement calibration reversed sign and is rejected."
            ),
        },
        "bounded_policy": {
            "reference": REFERENCE_POLICY,
            "fine": FINE_POLICY,
            "candidate": BOUNDED_POLICY,
            "rationale": (
                "Keep only the public-proven p>=0.55 gate and remove the locally "
                "detrimental 0.545<=p<0.55 boundary rows."
            ),
        },
        "locked_h2": {
            "candidate_changed_rows": int((bounded_gate & h2).sum()),
            "fine_changed_rows": int((fine_gate & h2).sum()),
            "candidate_vs_reference": bounded_vs_reference,
            "candidate_vs_fine": bounded_vs_fine,
            "candidate_vs_reference_monthly": monthly_vs_reference,
            "candidate_vs_reference_bootstrap": bootstrap,
        },
        "projection": {
            "locked_macro_score_gain_vs_fine": locked_macro_gain_vs_fine,
            "conservative_score_transfer_ratio": conservative_transfer,
            "projected_public_score_gain_vs_fine": projected_public_gain_vs_fine,
            "projected_public_score": public_fine["score"] + projected_public_gain_vs_fine,
            "target_gap_to_0_65": 0.65
            - (public_fine["score"] + projected_public_gain_vs_fine),
        },
        "submission": {
            "output": str(output_path) if promote else None,
            "created": promote,
            "changed_rows_vs_reference": int((movement > 1e-9).sum()),
            "changed_ratio_vs_reference": float((movement > 1e-9).mean()),
            "rows_different_from_fine": int((difference_from_fine > 1e-9).sum()),
            "mean_absolute_movement_kwh": float(movement.mean()),
            "max_absolute_movement_kwh": float(movement.max()),
            "promoted": promote,
            "promotion_blockers": []
            if promote
            else [
                "projected gain versus the public best is below the saturation margin",
                "locked bootstrap lower tail does not prove a positive incremental gain",
            ],
        },
        "decision": (
            "Promote the bounded candidate for submission."
            if promote
            else "Retain the bounded candidate for lineage; do not submit automatically."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz")
    parser.add_argument(
        "--pre-meta",
        default="submissions/archive/blend_best_crossg3_traj5_consensus.csv",
    )
    parser.add_argument(
        "--reference-meta", default="submissions/blend_best_crossg3_traj_meta25_p55.csv"
    )
    parser.add_argument(
        "--fine", default="submissions/blend_best_crossg3_traj_meta_finesweep.csv"
    )
    parser.add_argument(
        "--meta-sweep-report",
        default="artifacts_final/meta_gate_sweep/meta_gate_policy_sweep_report.json",
    )
    parser.add_argument(
        "--settlement-report",
        default="artifacts_final/calibration/settlement_composite_report.json",
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_meta_strong_p50_probe.csv"
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/meta_gate_sweep/bounded_public_transfer_report.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = run(
        Path(args.labels),
        Path(args.driver),
        Path(args.meta_cache),
        Path(args.pre_meta),
        Path(args.reference_meta),
        Path(args.fine),
        Path(args.meta_sweep_report),
        Path(args.settlement_report),
        Path(args.output),
        Path(args.report),
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
