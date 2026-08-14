from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exact_oof_meta_gate import (
    H2_START,
    META_SEEDS,
    Q2_START,
    _bootstrap_days,
    _evaluate_period,
    apply_meta_gate,
    fit_probabilities,
    settlement_benefit_labels,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from experiments.gefs_mean_disagreement_residual import build_disagreement_features


REFERENCE_THRESHOLD = 0.545
REFERENCE_ALPHA = 0.50


@dataclass(frozen=True)
class Policy:
    threshold: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return {"threshold": self.threshold, "alpha": self.alpha}


def compact_external_columns(frame: pd.DataFrame) -> list[str]:
    """Predeclared GEFS/GFS disagreement features; exclude raw absolute forecast levels."""
    prefixes = (
        "delta_u10_",
        "delta_v10_",
        "vector_disagreement_",
        "speed_disagreement_",
        "direction_cosine_",
    )
    columns = [column for column in frame.columns if column.startswith(prefixes)]
    if not columns:
        raise ValueError("No compact GEFS/GFS disagreement columns were found")
    return columns


def policies() -> tuple[Policy, ...]:
    # Kept deliberately small because this is a secondary gate, not another fine sweep.
    return tuple(
        Policy(threshold, alpha)
        for threshold in (0.50, 0.525, 0.55, 0.575, 0.60, 0.625)
        for alpha in (0.25, 0.375, 0.50)
    )


def _candidate(
    current: np.ndarray,
    member: np.ndarray,
    action: np.ndarray,
    probability: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_meta_gate(
        current,
        member,
        action,
        probability,
        threshold=policy.threshold,
        extra_alpha=policy.alpha,
    )


def compare_to_reference(
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    action: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    reference_seed_probabilities: list[np.ndarray],
    candidate_seed_probabilities: list[np.ndarray],
    period: np.ndarray,
    timestamps: pd.DatetimeIndex,
    policy: Policy,
) -> dict[str, object]:
    reference, reference_gate = _candidate(
        current,
        member,
        action,
        reference_probability,
        Policy(REFERENCE_THRESHOLD, REFERENCE_ALPHA),
    )
    candidate, candidate_gate = _candidate(
        current, member, action, candidate_probability, policy
    )
    comparison = _evaluate_period(truth, reference, candidate, period)
    seed_deltas: list[dict[str, float]] = []
    for reference_seed, candidate_seed in zip(
        reference_seed_probabilities, candidate_seed_probabilities, strict=True
    ):
        seed_reference, _ = _candidate(
            current,
            member,
            action,
            reference_seed,
            Policy(REFERENCE_THRESHOLD, REFERENCE_ALPHA),
        )
        seed_candidate, _ = _candidate(
            current, member, action, candidate_seed, policy
        )
        seed_deltas.append(
            _evaluate_period(truth, seed_reference, seed_candidate, period)["delta"]
        )
    monthly: dict[str, dict[str, float]] = {}
    for month in sorted(set(timestamps[period].month)):
        month_mask = period & (timestamps.month == month)
        if int(month_mask.sum()) >= 24:
            monthly[str(month)] = _evaluate_period(
                truth, reference, candidate, month_mask
            )["delta"]
    movement = candidate - reference
    return {
        "policy": policy.to_dict(),
        "reference_changed_rows": int((reference_gate & period).sum()),
        "candidate_changed_rows": int((candidate_gate & period).sum()),
        "incremental_changed_rows": int((np.abs(movement[period]) > 1e-9).sum()),
        "mean_absolute_incremental_movement_kwh": float(
            np.abs(movement[period]).mean()
        ),
        "comparison": comparison,
        "seed_deltas": seed_deltas,
        "min_seed_score_delta": float(
            min(delta["score"] for delta in seed_deltas)
        ),
        "all_seed_score_positive": bool(
            all(delta["score"] > 0.0 for delta in seed_deltas)
        ),
        "monthly_deltas": monthly,
        "positive_months": int(
            sum(delta["score"] > 0.0 for delta in monthly.values())
        ),
    }


def select_development(records: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = []
    for record in records:
        delta = record["comparison"]["delta"]
        if (
            delta["score"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and record["all_seed_score_positive"]
            and record["positive_months"] >= 2
        ):
            eligible.append(record)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            record["min_seed_score_delta"],
            record["comparison"]["delta"]["score"],
            -record["mean_absolute_incremental_movement_kwh"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--mean-features",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/enhanced_meta_gate.json",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    (
        _labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        meta,
        action,
    ) = _prepare_validation(Path(args.labels), Path(args.driver_cache))
    external = build_disagreement_features(Path(args.mean_features), Path(args.gfs))
    common = index.intersection(external.index)
    if len(common) < 0.999 * len(index):
        raise ValueError(
            f"GEFS features cover only {len(common)} of {len(index)} validation timestamps"
        )
    positions = index.get_indexer(common)
    if (positions < 0).any():
        raise AssertionError("GEFS/index alignment failed after intersection")
    index = common
    truth = truth[positions]
    member = member[positions]
    current = current[positions]
    meta = meta[positions]
    action = action[positions]
    external = external.reindex(index)
    external_columns = compact_external_columns(external)
    external_values = external[external_columns].to_numpy(dtype=float)
    if not np.isfinite(external_values).all():
        raise ValueError("GEFS disagreement matrix contains non-finite values")
    enhanced = np.column_stack([meta, external_values])

    q1 = np.asarray(index < Q2_START)
    q2 = np.asarray((index >= Q2_START) & (index < H2_START))
    h1 = np.asarray(index < H2_START)
    h2 = np.asarray(index >= H2_START)

    q1_labels = settlement_benefit_labels(truth, current, member, q1)
    q2_reference_probability, q2_reference_seeds = fit_probabilities(
        meta, q1_labels, q1 & action
    )
    q2_enhanced_probability, q2_enhanced_seeds = fit_probabilities(
        enhanced, q1_labels, q1 & action
    )
    development = [
        compare_to_reference(
            truth,
            current,
            member,
            action,
            q2_reference_probability,
            q2_enhanced_probability,
            q2_reference_seeds,
            q2_enhanced_seeds,
            q2,
            index,
            policy,
        )
        for policy in policies()
    ]
    selected = select_development(development)

    locked = None
    locked_bootstrap = None
    locked_opened = selected is not None
    if selected is not None:
        selected_policy = Policy(**selected["policy"])
        h1_labels = settlement_benefit_labels(truth, current, member, h1)
        h2_reference_probability, h2_reference_seeds = fit_probabilities(
            meta, h1_labels, h1 & action
        )
        h2_enhanced_probability, h2_enhanced_seeds = fit_probabilities(
            enhanced, h1_labels, h1 & action
        )
        locked = compare_to_reference(
            truth,
            current,
            member,
            action,
            h2_reference_probability,
            h2_enhanced_probability,
            h2_reference_seeds,
            h2_enhanced_seeds,
            h2,
            index,
            selected_policy,
        )
        reference, _ = _candidate(
            current,
            member,
            action,
            h2_reference_probability,
            Policy(REFERENCE_THRESHOLD, REFERENCE_ALPHA),
        )
        candidate, _ = _candidate(
            current,
            member,
            action,
            h2_enhanced_probability,
            selected_policy,
        )
        locked_bootstrap = _bootstrap_days(
            truth,
            reference,
            candidate,
            index,
            h2,
            args.bootstrap,
            seed=20260718,
        )

    locked_qualified = False
    if locked is not None and locked_bootstrap is not None:
        delta = locked["comparison"]["delta"]
        locked_qualified = bool(
            delta["score"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and locked["all_seed_score_positive"]
            and locked["positive_months"] >= 4
            and locked_bootstrap["positive_fraction"] >= 0.90
            and locked_bootstrap["q05"] >= 0.0
        )

    report = {
        "method": "GEFS/GFS disagreement-enhanced exact OOF meta gate",
        "hypothesis": (
            "Operational GEFS-vs-GFS disagreement helps veto unreliable cross-group "
            "settlement actions without directly predicting future residuals."
        ),
        "reference_policy": {
            "threshold": REFERENCE_THRESHOLD,
            "alpha": REFERENCE_ALPHA,
            "public_lineage": "blend_best_crossg3_traj_meta_finesweep.csv",
        },
        "validation_contract": {
            "development": "Q1 train -> Q2 select against fixed public-best gate",
            "locked": "H1 train -> H2 open only after Q2 all-component gate",
            "policy_count": len(policies()),
            "external_feature_count": len(external_columns),
            "external_columns": external_columns,
        },
        "development_selected": selected,
        "development_top": sorted(
            development,
            key=lambda item: item["comparison"]["delta"]["score"],
            reverse=True,
        )[:5],
        "locked_h2_opened": locked_opened,
        "locked_h2": locked,
        "locked_h2_bootstrap": locked_bootstrap,
        "qualified_for_2025_collection": locked_qualified,
        "decision": (
            "collect compliant 2025 GEFS and build candidate"
            if locked_qualified
            else "reject; do not collect 2025 GEFS or create submission"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
