"""Stress a local candidate under public/private-sized subset sampling.

DACON evaluates a pre-sampled 40% public subset and the complementary 60%
private subset.  Full-year and resampled-full-year validation can therefore
approve a factor whose discontinuous FiCR contribution is unstable on a
smaller subset.  This audit estimates that instability with both IID and
month-stratified complementary timestamp splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.group12_difference_reconciliation import (
    GROUPS,
    build_pairwise_features,
    pair_delta,
)
from experiments.group12_selective_gain_gate import _make_fixed_proposal
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _summarize(
    values: np.ndarray,
    observed: dict[str, float] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for position, component in enumerate(COMPONENTS):
        column = values[:, position]
        row: dict[str, Any] = {
            "mean": float(np.mean(column)),
            "standard_deviation": float(np.std(column, ddof=1)),
            "positive_fraction": float(np.mean(column > 0.0)),
            "q01": float(np.quantile(column, 0.01)),
            "q025": float(np.quantile(column, 0.025)),
            "q05": float(np.quantile(column, 0.05)),
            "median": float(np.quantile(column, 0.50)),
            "q95": float(np.quantile(column, 0.95)),
            "q975": float(np.quantile(column, 0.975)),
            "q99": float(np.quantile(column, 0.99)),
        }
        if observed is not None:
            target = float(observed[component])
            row["observed_public_pair_delta"] = target
            row["observed_percentile"] = float(np.mean(column <= target))
        output[component] = row
    return output


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def complementary_subset_stress(
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    *,
    public_fraction: float,
    repetitions: int,
    seed: int,
    stratify_month: bool,
    observed_public_pair_delta: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not 0.0 < public_fraction < 1.0:
        raise ValueError("public fraction must be within (0, 1)")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    public_values = np.empty((repetitions, len(COMPONENTS)), dtype=float)
    private_values = np.empty_like(public_values)
    positions = np.arange(len(index))
    strata = (
        [np.flatnonzero(index.month == month) for month in range(1, 13)]
        if stratify_month
        else [positions]
    )
    for iteration in range(repetitions):
        selected_parts: list[np.ndarray] = []
        for stratum in strata:
            count = max(1, int(round(public_fraction * len(stratum))))
            selected_parts.append(
                rng.choice(stratum, size=count, replace=False)
            )
        selected = np.concatenate(selected_parts)
        public_rows = np.zeros(len(index), dtype=bool)
        public_rows[selected] = True
        private_rows = ~public_rows
        public_delta = pair_delta(
            truth,
            reference,
            candidate,
            public_rows,
        )
        private_delta = pair_delta(
            truth,
            reference,
            candidate,
            private_rows,
        )
        public_values[iteration] = [
            public_delta[component] for component in COMPONENTS
        ]
        private_values[iteration] = [
            private_delta[component] for component in COMPONENTS
        ]
    correlation = {
        component: _safe_correlation(
            public_values[:, position],
            private_values[:, position],
        )
        for position, component in enumerate(COMPONENTS)
    }
    return {
        "contract": {
            "public_fraction": public_fraction,
            "private_fraction": 1.0 - public_fraction,
            "complementary_splits": True,
            "same_timestamp_subset_for_both_groups": True,
            "stratified_by_target_month": stratify_month,
            "sampling_without_replacement": True,
        },
        "repetitions": int(repetitions),
        "public": _summarize(
            public_values,
            observed_public_pair_delta,
        ),
        "private": _summarize(private_values, None),
        "public_private_delta_correlation": correlation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    baselines, truth_series, index, _issues = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    reference = {
        target: baselines[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    truth = {
        target: truth_series[target].to_numpy(dtype=float)
        for target in GROUPS
    }
    labels = pd.read_csv(_rooted(args.labels), encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").sort_index()
    pair_features = build_pairwise_features(
        pd.read_pickle(_rooted(args.feature_cache))
    )
    candidate, _unanimous = _make_fixed_proposal(
        pair_features,
        labels,
        index,
        reference,
        n_estimators=args.proposal_estimators,
    )
    observed_report = json.loads(
        _rooted(args.public_result).read_text(encoding="utf-8")
    )
    observed = observed_report["public_pair_delta"]
    full = pair_delta(
        truth,
        reference,
        candidate,
        np.ones(len(index), dtype=bool),
    )
    iid = complementary_subset_stress(
        truth,
        reference,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=args.repetitions,
        seed=20260729,
        stratify_month=False,
        observed_public_pair_delta=observed,
    )
    stratified = complementary_subset_stress(
        truth,
        reference,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=args.repetitions,
        seed=20260730,
        stratify_month=True,
        observed_public_pair_delta=observed,
    )
    report = {
        "family": "public_private_subset_stress",
        "official_split": {
            "public_fraction": 0.40,
            "private_fraction": 0.60,
            "sampling_scheme_disclosed": False,
        },
        "local_full_pair_delta": full,
        "observed_public_pair_delta": observed,
        "iid_timestamp_splits": iid,
        "month_stratified_timestamp_splits": stratified,
        "promotion_audit": {
            "old_full_year_positive": bool(
                all(full[component] >= 0.0 for component in COMPONENTS)
            ),
            "iid_public_q05_nonnegative": bool(
                all(
                    iid["public"][component]["q05"] >= 0.0
                    for component in COMPONENTS
                )
            ),
            "stratified_public_q05_nonnegative": bool(
                all(
                    stratified["public"][component]["q05"] >= 0.0
                    for component in COMPONENTS
                )
            ),
            "corrected_gate_passed": bool(
                all(
                    result["public"][component]["q05"] >= 0.0
                    and result["private"][component]["q05"] >= 0.0
                    for result in (iid, stratified)
                    for component in COMPONENTS
                )
            ),
        },
    }
    output = _rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache",
        default="artifacts_final/feature_cache/features_train.pkl",
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--public-result",
        default=(
            "artifacts_final/diagnostics/"
            "group12_reconciliation_public_result_20260728.json"
        ),
    )
    parser.add_argument("--proposal-estimators", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "public_private_subset_stress_20260729.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "local_full_pair_delta": report["local_full_pair_delta"],
                "iid_public": report["iid_timestamp_splits"]["public"],
                "stratified_public": (
                    report["month_stratified_timestamp_splits"]["public"]
                ),
                "promotion_audit": report["promotion_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
