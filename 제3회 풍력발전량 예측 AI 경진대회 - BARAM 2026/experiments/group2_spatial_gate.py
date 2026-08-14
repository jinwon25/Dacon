from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import assign_issue_blocks, load_issue_times
from experiments.cross_group_transfer import selected_prediction
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_2"
CAPACITY = CAPACITY_KWH[TARGET]
Q1_END = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 00:00:00")


@dataclass(frozen=True)
class GatePolicy:
    uncertainty_max: float
    distance_min: float
    alpha: float
    distance_max: float = 0.20

    @property
    def name(self) -> str:
        return (
            f"u{self.uncertainty_max:.2f}_d{self.distance_min:.2f}"
            f"_m{self.distance_max:.2f}_a{self.alpha:.2f}"
        )


def predefined_policies() -> list[GatePolicy]:
    """Return the small, fixed development grid.

    The grid contains only label-free gate quantities. It is intentionally
    constructed in code rather than expanded after looking at locked H2.
    """
    return [
        GatePolicy(uncertainty, distance, alpha)
        for uncertainty in (0.02, 0.04, 0.08)
        for distance in (0.04, 0.08)
        for alpha in (0.05, 0.10, 0.20)
    ]


def consensus_gate(
    base: np.ndarray,
    seed_17: np.ndarray,
    seed_29: np.ndarray,
    policy: GatePolicy,
) -> np.ndarray:
    """Select bounded rows using predictions only, never target values."""
    base = np.asarray(base, dtype=float)
    seed_17 = np.asarray(seed_17, dtype=float)
    seed_29 = np.asarray(seed_29, dtype=float)
    if base.shape != seed_17.shape or base.shape != seed_29.shape:
        raise ValueError("Base and seed predictions must have identical shapes")
    member = 0.5 * (seed_17 + seed_29)
    delta_17 = seed_17 - base
    delta_29 = seed_29 - base
    distance = np.abs(member - base) / CAPACITY
    uncertainty = np.abs(seed_17 - seed_29) / CAPACITY
    return (
        (delta_17 * delta_29 > 0.0)
        & (uncertainty <= policy.uncertainty_max)
        & (distance >= policy.distance_min)
        & (distance <= policy.distance_max)
        & np.isfinite(member)
        & np.isfinite(base)
    )


def apply_policy(base: np.ndarray, member: np.ndarray, gate: np.ndarray, alpha: float) -> np.ndarray:
    candidate = np.asarray(base, dtype=float).copy()
    candidate[gate] += alpha * (np.asarray(member, dtype=float)[gate] - candidate[gate])
    return np.clip(candidate, 0.0, CAPACITY)


def _metric_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    reference_metric = evaluate_group(truth[rows], reference[rows], CAPACITY)
    candidate_metric = evaluate_group(truth[rows], candidate[rows], CAPACITY)
    reference_values = reference_metric.to_dict()
    candidate_values = candidate_metric.to_dict()
    return {
        "rows": int(rows.sum()),
        "eligible_rows": int(candidate_metric.n_samples),
        "reference": reference_values,
        "candidate": candidate_values,
        "delta": {
            key: float(candidate_values[key] - reference_values[key])
            for key in ("score", "one_minus_nmae", "ficr")
        },
    }


def _component_values(metrics: dict[str, Any]) -> list[float]:
    return [float(metrics[period]["delta"][key]) for period in ("q1", "q2") for key in ("score", "one_minus_nmae", "ficr")]


def select_from_development(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select using Q1/Q2 fields only; locked confirmation is not accepted here."""
    eligible = [record for record in records if record["development_passed"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["development_robust_floor"]),
            float(record["development_mean_score_delta"]),
            -float(record["policy"]["alpha"]),
        ),
    )


def issue_cycle_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issue_times: pd.DatetimeIndex,
    strata: np.ndarray,
    rows: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap complete NWP issue cycles, stratified by issue-centre season."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rows = np.asarray(rows, dtype=bool)
    issue_values = np.asarray(issue_times)
    strata = np.asarray(strata)
    positions: dict[tuple[str, np.datetime64], np.ndarray] = {}
    stratum_issues: dict[str, np.ndarray] = {}
    for stratum in sorted(set(strata[rows])):
        issues = np.unique(issue_values[rows & (strata == stratum)])
        stratum_issues[stratum] = issues
        for issue in issues:
            positions[(stratum, issue)] = np.flatnonzero(
                rows & (strata == stratum) & (issue_values == issue)
            )
    if not stratum_issues:
        raise ValueError("Bootstrap rows are empty")
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_rows: list[np.ndarray] = []
        for stratum, issues in stratum_issues.items():
            sampled = rng.choice(issues, size=len(issues), replace=True)
            sampled_rows.extend(positions[(stratum, issue)] for issue in sampled)
        selected = np.concatenate(sampled_rows)
        values[iteration] = (
            evaluate_group(truth[selected], candidate[selected], CAPACITY).score
            - evaluate_group(truth[selected], reference[selected], CAPACITY).score
        )
    return {
        "n_bootstrap": int(n_bootstrap),
        "positive_fraction": float(np.mean(values > 0.0)),
        "q025": float(np.quantile(values, 0.025)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q975": float(np.quantile(values, 0.975)),
    }


def complete_issue_mask(
    full_issue_times: pd.DatetimeIndex,
    common_issue_times: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove whole issue cycles if cache alignment dropped any horizon row."""
    full_counts = pd.Series(1, index=np.asarray(full_issue_times)).groupby(level=0).sum()
    common_counts = pd.Series(1, index=np.asarray(common_issue_times)).groupby(level=0).sum()
    expected = int(full_counts.mode().iloc[0])
    complete_issues = common_counts[
        (common_counts == expected)
        & (full_counts.reindex(common_counts.index).fillna(0).astype(int) == expected)
    ].index
    mask = np.isin(np.asarray(common_issue_times), complete_issues)
    return mask, {
        "expected_rows_per_issue": expected,
        "full_issue_cycles": int(len(full_counts)),
        "retained_issue_cycles": int(len(complete_issues)),
        "dropped_incomplete_issue_cycles": int(len(common_counts) - len(complete_issues)),
        "dropped_rows_from_incomplete_cycles": int((~mask).sum()),
    }


def blocked_audit(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    timestamps: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    rows: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=bool)
    month, season = assign_issue_blocks(timestamps, issue_times)

    def evaluate_blocks(blocks: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for block in dict.fromkeys(blocks[rows]):
            selected = rows & (blocks == block)
            if int(selected.sum()) < 24:
                continue
            result[str(block)] = {
                "issue_cycles": int(pd.Index(issue_times[selected]).nunique()),
                **_metric_delta(truth, reference, candidate, selected),
            }
        return result

    monthly = evaluate_blocks(month)
    seasonal = evaluate_blocks(season)
    rolling = []
    ordered_seasons = list(seasonal)
    for position in range(1, len(ordered_seasons)):
        prior = rows & np.isin(season, ordered_seasons[:position])
        validation = rows & (season == ordered_seasons[position])
        overlap = set(issue_times[prior]).intersection(issue_times[validation])
        if overlap:
            raise AssertionError("An NWP issue cycle was split across rolling folds")
        rolling.append(
            {
                "train_seasons": ordered_seasons[:position],
                "validation_season": ordered_seasons[position],
                "issue_overlap": 0,
                "metrics": _metric_delta(truth, reference, candidate, validation),
            }
        )
    overall = _metric_delta(truth, reference, candidate, rows)
    bootstrap = issue_cycle_bootstrap(
        truth, reference, candidate, issue_times, season, rows, n_bootstrap, seed
    )
    monthly_deltas = [float(item["delta"]["score"]) for item in monthly.values()]
    seasonal_deltas = [float(item["delta"]["score"]) for item in seasonal.values()]
    passed = bool(
        min(float(overall["delta"][key]) for key in ("score", "one_minus_nmae", "ficr")) > 0.0
        and min(monthly_deltas) >= 0.0
        and min(seasonal_deltas) >= 0.0
        and float(bootstrap["q05"]) >= 0.0
        and float(bootstrap["positive_fraction"]) >= 0.90
    )
    return {
        "overall": overall,
        "monthly": monthly,
        "seasonal": seasonal,
        "rolling_folds": rolling,
        "issue_cycle_bootstrap": bootstrap,
        "worst_month_score_delta": min(monthly_deltas),
        "worst_season_score_delta": min(seasonal_deltas),
        "passed": passed,
    }


def run(
    seed17_path: Path,
    seed29_path: Path,
    base_cache_path: Path,
    issue_source_path: Path,
    output_path: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    seed17_cache = np.load(seed17_path)
    seed29_cache = np.load(seed29_path)
    seed17_index = pd.DatetimeIndex(pd.to_datetime(seed17_cache["timestamps_ns"]))
    seed29_index = pd.DatetimeIndex(pd.to_datetime(seed29_cache["timestamps_ns"]))
    if not seed17_index.equals(seed29_index):
        raise ValueError("Spatial seed validation timestamps differ")
    if not np.allclose(seed17_cache["truth"], seed29_cache["truth"], equal_nan=True):
        raise ValueError("Spatial seed validation truth arrays differ")

    base_cache = np.load(base_cache_path, allow_pickle=True)
    base_index = pd.DatetimeIndex(pd.to_datetime(base_cache[f"{TARGET}__valid_index_ns"]))
    common = seed17_index.intersection(base_index)
    spatial_position = seed17_index.get_indexer(common)
    base_position = base_index.get_indexer(common)
    if (spatial_position < 0).any() or (base_position < 0).any():
        raise ValueError("Failed to align spatial and weighted OOF caches")

    full_issue_times = load_issue_times(issue_source_path, seed17_index)
    issue_times = load_issue_times(issue_source_path, common)
    complete, issue_integrity = complete_issue_mask(full_issue_times, issue_times)
    truth = seed17_cache["truth"][spatial_position, 1].astype(float) * CAPACITY
    base = selected_prediction(base_cache, TARGET)[base_position]
    seed17 = seed17_cache["prediction"][spatial_position, 1].astype(float) * CAPACITY
    seed29 = seed29_cache["prediction"][spatial_position, 1].astype(float) * CAPACITY
    ensemble = 0.5 * (seed17 + seed29)
    eligible = truth >= 0.10 * CAPACITY
    split_masks = {
        "q1": complete & eligible & (common < Q1_END),
        "q2": complete & eligible & (common >= Q1_END) & (common < H2_START),
        "h2": complete & eligible & (common >= H2_START),
    }

    development_records: list[dict[str, Any]] = []
    for policy in predefined_policies():
        gate = consensus_gate(base, seed17, seed29, policy)
        candidates = {
            "ensemble": apply_policy(base, ensemble, gate, policy.alpha),
            "seed17": apply_policy(base, seed17, gate, policy.alpha),
            "seed29": apply_policy(base, seed29, gate, policy.alpha),
        }
        metrics = {
            member_name: {
                period: _metric_delta(truth, base, candidate, split_masks[period])
                for period in ("q1", "q2")
            }
            for member_name, candidate in candidates.items()
        }
        component_values = [
            float(metrics[member_name][period]["delta"][component])
            for member_name in ("ensemble", "seed17", "seed29")
            for period in ("q1", "q2")
            for component in ("score", "one_minus_nmae", "ficr")
        ]
        coverage = {
            period: float(np.mean(gate[split_masks[period]])) for period in ("q1", "q2")
        }
        development_records.append(
            {
                "policy": asdict(policy) | {"name": policy.name},
                "coverage": coverage,
                "metrics": metrics,
                "development_robust_floor": float(min(component_values)),
                "development_mean_score_delta": float(
                    np.mean(
                        [metrics["ensemble"][period]["delta"]["score"] for period in ("q1", "q2")]
                    )
                ),
                "development_passed": bool(
                    min(component_values) > 0.0 and min(coverage.values()) >= 0.01
                ),
            }
        )

    selected_record = select_from_development(development_records)
    locked_confirmation: dict[str, Any] | None = None
    development_bootstrap: dict[str, Any] | None = None
    promotion_passed = False
    if selected_record is not None:
        selected_policy = GatePolicy(
            uncertainty_max=float(selected_record["policy"]["uncertainty_max"]),
            distance_min=float(selected_record["policy"]["distance_min"]),
            distance_max=float(selected_record["policy"]["distance_max"]),
            alpha=float(selected_record["policy"]["alpha"]),
        )
        gate = consensus_gate(base, seed17, seed29, selected_policy)
        ensemble_candidate = apply_policy(base, ensemble, gate, selected_policy.alpha)
        _, season = assign_issue_blocks(common, issue_times)
        development_bootstrap = {
            period: issue_cycle_bootstrap(
                truth,
                base,
                ensemble_candidate,
                issue_times,
                season,
                split_masks[period],
                n_bootstrap,
                20260719 + position,
            )
            for position, period in enumerate(("q1", "q2"))
        }
        locked_confirmation = blocked_audit(
            truth,
            base,
            ensemble_candidate,
            common,
            issue_times,
            split_masks["h2"],
            n_bootstrap,
            20260721,
        )
        locked_confirmation["seed_stability"] = {
            member_name: _metric_delta(
                truth,
                base,
                apply_policy(base, member, gate, selected_policy.alpha),
                split_masks["h2"],
            )
            for member_name, member in (("seed17", seed17), ("seed29", seed29))
        }
        development_bootstrap_passed = all(
            float(item["q05"]) >= 0.0 and float(item["positive_fraction"]) >= 0.90
            for item in development_bootstrap.values()
        )
        locked_seed_passed = all(
            min(float(item["delta"][key]) for key in ("score", "one_minus_nmae", "ficr")) > 0.0
            for item in locked_confirmation["seed_stability"].values()
        )
        promotion_passed = bool(
            development_bootstrap_passed
            and locked_confirmation["passed"]
            and locked_seed_passed
        )
        locked_confirmation["development_bootstrap_passed"] = development_bootstrap_passed
        locked_confirmation["locked_seed_stability_passed"] = locked_seed_passed

    report: dict[str, Any] = {
        "method": "two-seed consensus bounded spatial blend for kpx_group_2",
        "selection_contract": {
            "development_only": "Q1 and Q2 metrics across ensemble, seed17, and seed29",
            "locked_confirmation": "H2 inspected exactly once after policy selection",
            "gate_inputs": [
                "seed direction agreement",
                "seed disagreement divided by capacity",
                "ensemble-to-base distance divided by capacity",
            ],
            "promotion_requirements": [
                "all Q1/Q2 score, 1-NMAE, and FICR deltas positive for both seeds and ensemble",
                "Q1 and Q2 issue-cycle bootstrap q05 non-negative and positive fraction >= 0.90",
                "H2 overall components, every month, every season, and issue bootstrap pass",
                "both individual seeds have positive H2 component deltas",
            ],
        },
        "alignment": {
            "spatial_rows": int(len(seed17_index)),
            "base_rows": int(len(base_index)),
            "common_rows": int(len(common)),
            **issue_integrity,
        },
        "predefined_policy_count": len(development_records),
        "development_records": development_records,
        "selected_from_q1_q2": selected_record,
        "development_issue_bootstrap": development_bootstrap,
        "locked_h2_confirmation": locked_confirmation,
        "promotion": {
            "passed": promotion_passed,
            "submission_candidate_created": False,
            "reason": (
                "Strict group-2 spatial gate passed; final-model prediction is required before file creation."
                if promotion_passed
                else "Strict stability contract failed; do not create a submission candidate."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed17",
        default="artifacts_final/spatiotemporal/validation_predictions.npz",
    )
    parser.add_argument(
        "--seed29",
        default="artifacts_final/spatiotemporal_seed29/validation_predictions.npz",
    )
    parser.add_argument(
        "--base-cache",
        default="artifacts_final/lineage_inputs/weighted_prediction_cache.npz",
    )
    parser.add_argument("--issue-source", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output",
        default="artifacts_final/group2_spatial_gate/group2_spatial_gate_report.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    report = run(
        Path(args.seed17),
        Path(args.seed29),
        Path(args.base_cache),
        Path(args.issue_source),
        Path(args.output),
        args.n_bootstrap,
    )
    print(
        json.dumps(
            {
                "selected": report["selected_from_q1_q2"],
                "locked_h2": report["locked_h2_confirmation"],
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
