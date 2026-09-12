from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.kma_um_power_curve_gate import (
    CAPACITY,
    COMPONENTS,
    TARGET,
    _compare,
    issue_block_bootstrap,
)
from experiments.ldaps_neighborhood_decision import (
    _month_deltas,
    _seasonal_deltas,
)
from experiments.ldaps_upwind_advection_power_curve import (
    fit_power_curve,
    load_ldaps_hub_vectors,
    upwind_kernel_speed,
)


Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")


@dataclass(frozen=True)
class SmoothingPolicy:
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_smoothing_difference_kwh: float
    alpha: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def smooth_within_issue(
    values: np.ndarray,
    issue_times: np.ndarray,
) -> np.ndarray:
    """Apply the paper's centered ±1 h moving average within each issue."""
    values = np.asarray(values, dtype=float)
    issue_times = np.asarray(issue_times)
    if values.ndim != 1 or len(values) != len(issue_times):
        raise ValueError("values and issue times must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("smoothing input contains non-finite values")
    if pd.isna(issue_times).any():
        raise ValueError("smoothing issue times contain missing values")
    codes, unique = pd.factorize(issue_times, sort=False)
    output = values.copy()
    for code in range(len(unique)):
        positions = np.flatnonzero(codes == code)
        if len(positions) == 1:
            continue
        if not np.all(np.diff(positions) == 1):
            raise ValueError("one issue cycle appears in disjoint row blocks")
        local = values[positions]
        padded = np.pad(local, (1, 1), mode="edge")
        output[positions] = (
            padded[:-2] + padded[1:-1] + padded[2:]
        ) / 3.0
    return output


def smoothing_gate(
    raw_direct: np.ndarray,
    smoothed_direct: np.ndarray,
    incumbent: np.ndarray,
    available: np.ndarray,
    policy: SmoothingPolicy,
) -> np.ndarray:
    difference = smoothed_direct - raw_direct
    if policy.direction == "up":
        direction = difference > 0.0
    elif policy.direction == "down":
        direction = difference < 0.0
    elif policy.direction == "both":
        direction = difference != 0.0
    else:
        raise ValueError(f"unknown direction: {policy.direction}")
    ratio = incumbent / CAPACITY
    return (
        available
        & direction
        & (
            np.abs(difference)
            >= policy.minimum_smoothing_difference_kwh
        )
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )


def apply_smoothing_policy(
    raw_direct: np.ndarray,
    smoothed_direct: np.ndarray,
    incumbent: np.ndarray,
    available: np.ndarray,
    policy: SmoothingPolicy,
    *,
    maximum_incremental_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < maximum_incremental_movement_ratio <= 0.01:
        raise ValueError("incremental movement ratio must be in (0, 0.01]")
    gate = smoothing_gate(
        raw_direct,
        smoothed_direct,
        incumbent,
        available,
        policy,
    )
    bound = maximum_incremental_movement_ratio * CAPACITY
    movement = np.clip(
        policy.alpha * (smoothed_direct - raw_direct),
        -bound,
        bound,
    )
    candidate = incumbent.copy()
    candidate[gate] = incumbent[gate] + movement[gate]
    return np.clip(candidate, 0.0, CAPACITY), gate


def _development_month_deltas(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    selection: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for month in (4, 5, 6):
        mask = selection & np.asarray(index.month == month)
        if int(mask.sum()) >= 24:
            output[str(month)] = _compare(
                truth, incumbent, candidate, mask
            )["delta"]
    return output


def select_smoothing_policy(
    truth: np.ndarray,
    raw_direct: np.ndarray,
    smoothed_direct: np.ndarray,
    incumbent: np.ndarray,
    available: np.ndarray,
    index: pd.DatetimeIndex,
    selection: np.ndarray,
    *,
    minimum_changed_rows: int,
    maximum_changed_ratio: float,
    maximum_incremental_movement_ratio: float,
) -> dict[str, object]:
    difference = smoothed_direct - raw_direct
    screened: list[dict[str, object]] = []
    overall_positive: list[dict[str, object]] = []
    robust: list[dict[str, object]] = []
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = difference > 0.0
        elif direction_name == "down":
            direction = difference < 0.0
        else:
            direction = difference != 0.0
        pool_mask = selection & available & direction
        if not pool_mask.any():
            continue
        pool = np.abs(difference[pool_mask])
        for coverage in (0.10, 0.05):
            threshold = float(np.quantile(pool, 1.0 - coverage))
            for minimum_ratio, maximum_ratio in (
                (0.10, 0.80),
                (0.20, 0.80),
            ):
                for alpha in (0.10, 0.25, 0.50, 1.00):
                    policy = SmoothingPolicy(
                        direction=direction_name,
                        coverage=coverage,
                        minimum_base_ratio=minimum_ratio,
                        maximum_base_ratio=maximum_ratio,
                        minimum_smoothing_difference_kwh=threshold,
                        alpha=alpha,
                    )
                    candidate, gate = apply_smoothing_policy(
                        raw_direct,
                        smoothed_direct,
                        incumbent,
                        available,
                        policy,
                        maximum_incremental_movement_ratio=(
                            maximum_incremental_movement_ratio
                        ),
                    )
                    changed = (
                        selection
                        & gate
                        & (np.abs(candidate - incumbent) > 1e-9)
                    )
                    changed_rows = int(changed.sum())
                    if changed_rows < minimum_changed_rows:
                        continue
                    if (
                        changed_rows / int(selection.sum())
                        > maximum_changed_ratio
                    ):
                        continue
                    comparison = _compare(
                        truth, incumbent, candidate, selection
                    )
                    monthly = _development_month_deltas(
                        truth,
                        incumbent,
                        candidate,
                        index,
                        selection,
                    )
                    record = {
                        "policy": policy,
                        "comparison": comparison,
                        "development_monthly": monthly,
                        "changed_rows": changed_rows,
                        "changed_ratio": float(
                            changed_rows / int(selection.sum())
                        ),
                        "mean_absolute_increment_kwh": float(
                            np.mean(
                                np.abs(
                                    candidate[selection]
                                    - incumbent[selection]
                                )
                            )
                        ),
                    }
                    screened.append(record)
                    overall_pass = (
                        min(
                            comparison["delta"][component]
                            for component in COMPONENTS
                        )
                        > 0.0
                    )
                    if overall_pass:
                        overall_positive.append(record)
                    monthly_pass = bool(monthly) and all(
                        min(value[component] for component in COMPONENTS)
                        >= 0.0
                        for value in monthly.values()
                    )
                    if overall_pass and monthly_pass:
                        robust.append(record)

    def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
        delta = row["comparison"]["delta"]
        return (
            float(delta["score"]),
            float(min(delta["one_minus_nmae"], delta["ficr"])),
            -float(row["mean_absolute_increment_kwh"]),
        )

    screened.sort(key=sort_key, reverse=True)
    overall_positive.sort(key=sort_key, reverse=True)
    robust.sort(key=sort_key, reverse=True)

    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {**row, "policy": row["policy"].to_dict()}

    return {
        "selected": None if not robust else robust[0],
        "top_robust": [serialize(row) for row in robust[:10]],
        "top_overall_positive": [
            serialize(row) for row in overall_positive[:10]
        ],
        "top_screened": [serialize(row) for row in screened[:10]],
        "screened_policies": len(screened),
        "overall_positive_policies": len(overall_positive),
        "robust_policies": len(robust),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_submission(
    *,
    train_raw_speed: np.ndarray,
    train_smoothed_speed: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    policy: SmoothingPolicy,
    maximum_incremental_movement_ratio: float,
    test_ldaps: Path,
    incumbent_submission: Path,
    output_submission: Path,
) -> dict[str, object]:
    incumbent_frame = pd.read_csv(
        incumbent_submission, encoding="utf-8-sig"
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(incumbent_frame["forecast_kst_dtm"])
    )
    test_u, test_v, grid_x, grid_y, test_issues = (
        load_ldaps_hub_vectors(test_ldaps, index)
    )
    test_raw_speed = upwind_kernel_speed(
        test_u, test_v, grid_x, grid_y, 0.0
    )
    test_smoothed_speed = smooth_within_issue(
        test_raw_speed, test_issues.to_numpy()
    )
    raw_direct = fit_power_curve(
        train_raw_speed, truth, train_mask, test_raw_speed
    )
    smoothed_direct = fit_power_curve(
        train_smoothed_speed, truth, train_mask, test_smoothed_speed
    )
    incumbent = incumbent_frame[TARGET].to_numpy(dtype=float)
    candidate, gate = apply_smoothing_policy(
        raw_direct,
        smoothed_direct,
        incumbent,
        np.ones(len(index), dtype=bool),
        policy,
        maximum_incremental_movement_ratio=(
            maximum_incremental_movement_ratio
        ),
    )
    output = incumbent_frame.copy()
    output[TARGET] = candidate
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_submission, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(Path.cwd())).audit(
        output_submission
    )
    if not audit.valid:
        raise RuntimeError(f"candidate validation failed: {audit.errors}")
    if not output[
        ["kpx_group_1", "kpx_group_2"]
    ].equals(incumbent_frame[["kpx_group_1", "kpx_group_2"]]):
        raise RuntimeError("group-1 or group-2 changed unexpectedly")
    changed = np.abs(candidate - incumbent) > 1e-9
    return {
        "file": str(output_submission),
        "sha256": _sha256(output_submission),
        "audit": audit.to_dict(),
        "changed_rows": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "gate_rows": int(gate.sum()),
        "maximum_absolute_increment_kwh": float(
            np.max(np.abs(candidate - incumbent))
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    cache = np.load(args.power_curve_oof)
    index = pd.DatetimeIndex(pd.to_datetime(cache["index_ns"]))
    truth = cache["truth"].astype(float)
    cached_issues = pd.Series(pd.to_datetime(cache["issue_ns"]))
    static_incumbent = cache["static_candidate"].astype(float)
    rolling_incumbent = cache["rolling_candidate"].astype(float)
    available = cache["available"].astype(bool)
    q1 = (index < Q2_START) & available
    q2 = (index >= Q2_START) & (index < H2_START) & available
    h1 = (index < H2_START) & available
    h2 = (index >= H2_START) & (index < END) & available
    if not np.array_equal(q2, cache["q2"].astype(bool)):
        raise ValueError("Q2 split does not match the locked OOF cache")
    if not np.array_equal(h2, cache["h2"].astype(bool)):
        raise ValueError("H2 split does not match the locked OOF cache")

    hub_u, hub_v, grid_x, grid_y, ldaps_issues = (
        load_ldaps_hub_vectors(Path(args.train_ldaps), index)
    )
    source_issues = pd.Series(
        pd.to_datetime(ldaps_issues.to_numpy())
    ).reset_index(drop=True)
    known_issue = cached_issues.notna().to_numpy()
    issue_error_hours = float(
        np.max(
            np.abs(
                (
                    source_issues.to_numpy()[known_issue]
                    - cached_issues.to_numpy()[known_issue]
                )
                / np.timedelta64(1, "h")
            )
        )
    )
    if issue_error_hours > 0.0:
        raise ValueError("LDAPS issue times do not match the OOF cache")
    missing_cached_issues = int(cached_issues.isna().sum())
    issues = cached_issues.fillna(source_issues).to_numpy()
    if pd.isna(issues).any():
        raise ValueError("effective issue times remain incomplete")
    raw_speed = upwind_kernel_speed(
        hub_u, hub_v, grid_x, grid_y, 0.0
    )
    smoothed_speed = smooth_within_issue(raw_speed, issues)
    q1_raw_direct = fit_power_curve(
        raw_speed, truth, q1, raw_speed
    )
    q1_smoothed_direct = fit_power_curve(
        smoothed_speed, truth, q1, smoothed_speed
    )
    selection = select_smoothing_policy(
        truth,
        q1_raw_direct,
        q1_smoothed_direct,
        static_incumbent,
        available,
        index,
        q2,
        minimum_changed_rows=args.minimum_selection_rows,
        maximum_changed_ratio=args.maximum_changed_ratio,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    report: dict[str, Any] = {
        "family": "ldaps_temporal_wind_smoothing",
        "method": {
            "wind_field": (
                "group-3-centered Gaussian average of 117 m LDAPS vectors"
            ),
            "smoothing": (
                "centered moving average over forecast leads t-1, t, t+1 "
                "within each issue cycle"
            ),
            "edge_handling": "replicate the first/last lead within an issue",
            "power_mapping": (
                "separate monotone isotonic curves for raw and smoothed wind"
            ),
            "increment": (
                "blend only the smoothed-minus-raw power difference into "
                "the incumbent"
            ),
        },
        "multi_period_feasibility": {
            "requested": "causal exact 2023+2024 incumbent OOF",
            "available_group3_labels": {
                "2022": 0,
                "2023": 8759,
                "2024": 8778,
            },
            "feasible": False,
            "reason": (
                "A causal 2023 group-3 fold cannot be trained because 2022 "
                "contains no group-3 labels; the current cross-group lineage "
                "also uses 2023 group-3 labels."
            ),
            "fallback": (
                "2024 Q1 model train, Q2 robust selection by month, and one "
                "H1-refit locked H2 evaluation"
            ),
        },
        "contract": {
            "q1_model_train": "2024-01-01 through 2024-03-31",
            "q2_policy_selection": (
                "2024-04-01 through 2024-07-01 00:00"
            ),
            "q2_robustness": (
                "all score components nonnegative in April, May, and June"
            ),
            "locked_h2": (
                "H1-refit; 2024-07-01 01:00 through 2024-12-31 23:00"
            ),
            "maximum_changed_ratio": args.maximum_changed_ratio,
            "maximum_incremental_movement_ratio": (
                args.maximum_incremental_movement_ratio
            ),
            "public_score_used_for_selection": False,
        },
        "data": {
            "train_ldaps": args.train_ldaps,
            "power_curve_oof": args.power_curve_oof,
            "rows": len(index),
            "q1_rows": int(q1.sum()),
            "q2_rows": int(q2.sum()),
            "h1_rows": int(h1.sum()),
            "h2_rows": int(h2.sum()),
            "issue_time_max_error_hours": issue_error_hours,
            "cached_issue_times_filled_from_ldaps": (
                missing_cached_issues
            ),
            "raw_smoothed_speed_mae": float(
                np.mean(np.abs(smoothed_speed - raw_speed))
            ),
        },
        "selection": {
            **selection,
            "selected": (
                None
                if selection["selected"] is None
                else {
                    **selection["selected"],
                    "policy": selection["selected"]["policy"].to_dict(),
                }
            ),
        },
    }
    if selection["selected"] is None:
        report["promotion_gates"] = {
            "q2_overall_all_components_positive": (
                selection["overall_positive_policies"] > 0
            ),
            "q2_all_months_all_components_nonnegative": False,
        }
        report["decision"] = {
            "tier": "rejected",
            "submission_requested": bool(
                args.write_submission_if_qualified
            ),
            "submission_created": False,
            "submission_eligible": False,
            "reason": (
                "no temporal-wind-smoothing policy passed the Q2 monthly "
                "robustness contract"
            ),
        }
        report["submission"] = None
        _write_json(Path(args.output), report)
        return report

    policy = selection["selected"]["policy"]
    static_candidate, static_gate = apply_smoothing_policy(
        q1_raw_direct,
        q1_smoothed_direct,
        static_incumbent,
        available,
        policy,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    static_locked = _compare(
        truth, static_incumbent, static_candidate, h2
    )
    h1_raw_direct = fit_power_curve(
        raw_speed, truth, h1, raw_speed
    )
    h1_smoothed_direct = fit_power_curve(
        smoothed_speed, truth, h1, smoothed_speed
    )
    locked_candidate, locked_gate = apply_smoothing_policy(
        h1_raw_direct,
        h1_smoothed_direct,
        rolling_incumbent,
        available,
        policy,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    locked = _compare(
        truth, rolling_incumbent, locked_candidate, h2
    )
    monthly = _month_deltas(
        truth, rolling_incumbent, locked_candidate, index, h2
    )
    seasonal = _seasonal_deltas(
        truth, rolling_incumbent, locked_candidate, index, h2
    )
    bootstrap = issue_block_bootstrap(
        truth,
        rolling_incumbent,
        locked_candidate,
        issues,
        h2,
        n_bootstrap=args.n_bootstrap,
        seed=20260726,
    )
    movement = locked_candidate - rolling_incumbent
    changed_h2 = h2 & locked_gate & (np.abs(movement) > 1e-9)
    gates = {
        "q2_overall_all_components_positive": min(
            selection["selected"]["comparison"]["delta"][component]
            for component in COMPONENTS
        )
        > 0.0,
        "q2_all_months_all_components_nonnegative": all(
            min(value[component] for component in COMPONENTS) >= 0.0
            for value in selection["selected"][
                "development_monthly"
            ].values()
        ),
        "static_q1_curve_h2_all_components_positive": min(
            static_locked["delta"][component] for component in COMPONENTS
        )
        > 0.0,
        "rolling_h1_curve_h2_all_components_positive": min(
            locked["delta"][component] for component in COMPONENTS
        )
        > 0.0,
        "all_locked_seasonal_components_nonnegative": all(
            min(value[component] for component in COMPONENTS) >= 0.0
            for value in seasonal.values()
        ),
        "all_locked_month_scores_nonnegative": all(
            value["score"] >= 0.0 for value in monthly.values()
        ),
        "all_locked_month_ficr_nonnegative": all(
            value["ficr"] >= 0.0 for value in monthly.values()
        ),
        "bootstrap_q05_all_components_positive": min(
            bootstrap["q05"].values()
        )
        > 0.0,
        "bootstrap_positive_fraction_passed": bootstrap[
            "positive_all_component_fraction"
        ]
        >= args.minimum_bootstrap_positive_fraction,
        "minimum_locked_group3_score_gain_passed": locked["delta"]["score"]
        >= args.minimum_locked_group3_score_gain,
        "maximum_changed_h2_ratio_passed": (
            int(changed_h2.sum()) / int(h2.sum())
        )
        <= args.maximum_changed_ratio,
        "maximum_incremental_movement_passed": float(
            np.max(np.abs(movement[h2]))
        )
        <= args.maximum_incremental_movement_ratio * CAPACITY + 1e-9,
    }
    qualified = bool(all(gates.values()))
    submission = None
    if qualified and args.write_submission_if_qualified:
        submission = _write_submission(
            train_raw_speed=raw_speed,
            train_smoothed_speed=smoothed_speed,
            truth=truth,
            train_mask=(index < END) & available,
            policy=policy,
            maximum_incremental_movement_ratio=(
                args.maximum_incremental_movement_ratio
            ),
            test_ldaps=Path(args.test_ldaps),
            incumbent_submission=Path(args.incumbent_submission),
            output_submission=Path(args.output_submission),
        )
    report["selected_policy"] = policy.to_dict()
    report["locked"] = {
        "static_q1_curve_h2": static_locked,
        "rolling_h1_curve_h2": locked,
        "seasonal": seasonal,
        "monthly": monthly,
        "bootstrap": bootstrap,
        "changed_rows": int(changed_h2.sum()),
        "changed_ratio": float(changed_h2.sum() / int(h2.sum())),
        "maximum_absolute_increment_kwh": float(
            np.max(np.abs(movement[h2]))
        ),
        "mean_absolute_increment_kwh": float(
            np.mean(np.abs(movement[h2]))
        ),
        "static_changed_rows": int((static_gate & h2).sum()),
    }
    report["promotion_gates"] = gates
    report["decision"] = {
        "tier": "candidate" if qualified else "rejected",
        "submission_requested": bool(args.write_submission_if_qualified),
        "submission_created": submission is not None,
        "submission_eligible": submission is not None and qualified,
        "reason": (
            "temporal-wind-smoothing candidate passed every locked gate"
            if qualified
            else "temporal-wind-smoothing candidate failed a locked gate"
        ),
    }
    report["submission"] = submission
    _write_json(Path(args.output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ldaps", default="data/train/ldaps_train.csv"
    )
    parser.add_argument(
        "--test-ldaps", default="data/test/ldaps_test.csv"
    )
    parser.add_argument(
        "--power-curve-oof",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument("--minimum-selection-rows", type=int, default=24)
    parser.add_argument(
        "--maximum-changed-ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--maximum-incremental-movement-ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--minimum-locked-group3-score-gain",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--minimum-bootstrap-positive-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/diagnostics/"
            "ldaps_temporal_wind_smoothing_20260726.json"
        ),
    )
    parser.add_argument(
        "--incumbent-submission",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument(
        "--output-submission",
        default=(
            "artifacts_final/candidates/"
            "ldaps_temporal_wind_smoothing_20260726.csv"
        ),
    )
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "family": report["family"],
                "selected_policy": report.get("selected_policy"),
                "locked": report.get("locked"),
                "promotion_gates": report["promotion_gates"],
                "decision": report["decision"],
                "submission": report["submission"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
