"""Frozen group-3 spatial/temporal displacement experiment.

Selection is restricted to two 2023 time-forward folds.  Only after one
candidate family and overlay weight are frozen is the exact 2024 incumbent
group-3 OOF surface evaluated.  Groups 1 and 2 are always copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.audit_current_incumbent_oof import G1_WEIGHT, G2_WEIGHT
from experiments.compose_residual_stack_candidate import apply_capped_residual_stack
from experiments.kma_um_power_curve_gate import load_context_speed
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.ldaps_upwind_advection_power_curve import (
    fit_power_curve,
    load_ldaps_hub_vectors,
    upwind_kernel_speed,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_competition, evaluate_group
from src.sprint066_pipeline import validate_submission


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
COMPONENTS = ("score", "one_minus_nmae", "ficr")
WEIGHTS = (0.10, 0.25, 0.50)
MAXIMUM_MOVEMENT_RATIO = 0.01
EXPECTED_HASHES = {
    "candidate": "8056206176d12f21f72fda14fba8fa3b19bcc8a6da7d8a6e389b9e28a40c4902",
    "primary_cache": "0680805255858a65037fb3e21d2054a32fde33a2926cefd1909123a922c51056",
    "residual_cache": "540abbb0cb8a67330b3b8fbae0c31a6f33f29c1d1d1ddcd4ceee7d349bdf6003",
    "group3_cache": "9fc2e6a2f178e4c4fd4d3b15cc87ec9728b8601e925bb6bee01b0702c380443a",
}


@dataclass(frozen=True)
class DisplacementSpec:
    candidate_id: str
    description: str


@dataclass(frozen=True)
class FrozenRule:
    candidate_id: str
    weight: float
    movement_cap_ratio: float = MAXIMUM_MOVEMENT_RATIO


CANDIDATES = (
    DisplacementSpec(
        "d1_nearest_grid",
        "nearest LDAPS grid to the documented group-3 turbine centroid",
    ),
    DisplacementSpec(
        "d2_upstream_1p5km",
        "instantaneous 1.5 km upstream Gaussian displacement",
    ),
    DisplacementSpec(
        "d3_upstream_3p0km",
        "instantaneous 3.0 km upstream Gaussian displacement",
    ),
    DisplacementSpec(
        "d4_temporal_minus1h",
        "site-centred speed from the previous lead inside the same issuance",
    ),
    DisplacementSpec(
        "d5_temporal_plus1h",
        "site-centred speed from the next lead inside the same issuance",
    ),
    DisplacementSpec(
        "d6_upstream_1p5km_mean_pm1h",
        "1.5 km upstream speed followed by centred +/-1 h issue-local mean",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Forecast day is 01:00 through the following 00:00 KST."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(hours=1)).normalize()


def prior_day_cutoffs(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return target_dates(index) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)


def audit_availability(
    index: pd.DatetimeIndex,
    issues: np.ndarray | pd.Series | pd.DatetimeIndex,
    source: str,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(index)
    issue_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(issues)))
    if len(index) != len(issue_index):
        raise ValueError(f"{source} index and issue rows differ")
    if index.has_duplicates:
        raise ValueError(f"{source} forecast timestamps contain duplicates")
    if issue_index.isna().any():
        raise ValueError(f"{source} issue timestamps contain missing values")
    cutoffs = prior_day_cutoffs(index)
    violation = issue_index > cutoffs
    if violation.any():
        position = int(np.flatnonzero(violation)[0])
        raise AssertionError(
            f"{source} availability after cutoff at {index[position]}: "
            f"{issue_index[position]} > {cutoffs[position]}"
        )
    lead_hours = (index - issue_index).total_seconds().to_numpy(dtype=float) / 3600.0
    if not np.isfinite(lead_hours).all() or np.any(lead_hours < 0.0):
        raise AssertionError(f"{source} contains invalid actual lead hours")
    return {
        "source": source,
        "rows": int(len(index)),
        "duplicate_forecast_rows": 0,
        "cutoff_violations": 0,
        "issue_cycles": int(issue_index.nunique()),
        "lead_hour_min": float(lead_hours.min()),
        "lead_hour_max": float(lead_hours.max()),
        "maximum_issue_minus_cutoff_hours": float(
            ((issue_index - cutoffs).total_seconds() / 3600.0).max()
        ),
    }


def shift_within_issue(
    values: np.ndarray,
    issue_times: np.ndarray | pd.Series | pd.DatetimeIndex,
    offset: int,
) -> np.ndarray:
    """Shift an hourly vector without ever crossing an issuance boundary."""
    values = np.asarray(values, dtype=float)
    issues = np.asarray(pd.to_datetime(np.asarray(issue_times)))
    if values.ndim != 1 or len(values) != len(issues):
        raise ValueError("values and issue times must be aligned vectors")
    if not np.isfinite(values).all() or pd.isna(issues).any():
        raise ValueError("issue-local shift inputs must be finite")
    output = values.copy()
    codes, unique = pd.factorize(issues, sort=False)
    for code in range(len(unique)):
        positions = np.flatnonzero(codes == code)
        if len(positions) > 1 and not np.all(np.diff(positions) == 1):
            raise ValueError("one issuance appears in disjoint row blocks")
        local_positions = np.clip(np.arange(len(positions)) + offset, 0, len(positions) - 1)
        output[positions] = values[positions[local_positions]]
    return output


def mean_pm1_within_issue(
    values: np.ndarray,
    issue_times: np.ndarray | pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    previous = shift_within_issue(values, issue_times, -1)
    following = shift_within_issue(values, issue_times, 1)
    return (previous + np.asarray(values, dtype=float) + following) / 3.0


def displacement_speeds(
    hub_u: np.ndarray,
    hub_v: np.ndarray,
    grid_x_km: np.ndarray,
    grid_y_km: np.ndarray,
    issue_times: np.ndarray | pd.Series | pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    center = upwind_kernel_speed(
        hub_u, hub_v, grid_x_km, grid_y_km, 0.0, sigma_km=1.5
    )
    nearest_index = int(np.argmin(grid_x_km**2 + grid_y_km**2))
    nearest = np.hypot(hub_u[:, nearest_index], hub_v[:, nearest_index])
    upstream_1p5 = upwind_kernel_speed(
        hub_u, hub_v, grid_x_km, grid_y_km, 1.5, sigma_km=1.5
    )
    upstream_3p0 = upwind_kernel_speed(
        hub_u, hub_v, grid_x_km, grid_y_km, 3.0, sigma_km=1.5
    )
    variants = {
        "d1_nearest_grid": nearest,
        "d2_upstream_1p5km": upstream_1p5,
        "d3_upstream_3p0km": upstream_3p0,
        "d4_temporal_minus1h": shift_within_issue(center, issue_times, -1),
        "d5_temporal_plus1h": shift_within_issue(center, issue_times, 1),
        "d6_upstream_1p5km_mean_pm1h": mean_pm1_within_issue(
            upstream_1p5, issue_times
        ),
    }
    if tuple(variants) != tuple(spec.candidate_id for spec in CANDIDATES):
        raise AssertionError("runtime candidates differ from preregistration")
    for name, values in variants.items():
        if values.shape != center.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid displacement values for {name}")
    return center, variants


def apply_overlay(
    incumbent: np.ndarray,
    centered_member: np.ndarray,
    displaced_member: np.ndarray,
    rule: FrozenRule,
) -> np.ndarray:
    incumbent = np.asarray(incumbent, dtype=float)
    centered_member = np.asarray(centered_member, dtype=float)
    displaced_member = np.asarray(displaced_member, dtype=float)
    if incumbent.shape != centered_member.shape or incumbent.shape != displaced_member.shape:
        raise ValueError("overlay arrays must have identical shapes")
    if rule.candidate_id not in {spec.candidate_id for spec in CANDIDATES}:
        raise ValueError("overlay candidate was not preregistered")
    if rule.weight not in WEIGHTS:
        raise ValueError("overlay weight was not preregistered")
    bound = rule.movement_cap_ratio * CAPACITY
    movement = np.clip(rule.weight * (displaced_member - centered_member), -bound, bound)
    return np.clip(incumbent + movement, 0.0, CAPACITY)


def metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def compare_group(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    before = evaluate_group(truth[mask], incumbent[mask], CAPACITY)
    after = evaluate_group(truth[mask], candidate[mask], CAPACITY)
    return {
        "incumbent": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": metric_delta(before, after),
        "rows": int(mask.sum()),
        "changed_rows": int(np.sum(mask & (np.abs(candidate - incumbent) > 1e-9))),
    }


def group_row_masks(rows_per_group: dict[str, int]) -> dict[str, np.ndarray]:
    if tuple(rows_per_group) != tuple(CAPACITY_KWH):
        raise ValueError("group order must match official metric order")
    if any(rows < 1 for rows in rows_per_group.values()):
        raise ValueError("each group must contain at least one row")
    total = int(sum(rows_per_group.values()))
    masks: dict[str, np.ndarray] = {}
    start = 0
    for group, rows in rows_per_group.items():
        mask = np.zeros(total, dtype=bool)
        mask[start : start + rows] = True
        masks[group] = mask
        start += rows
    stacked = np.column_stack(list(masks.values()))
    if not np.all(stacked.sum(axis=1) == 1):
        raise AssertionError("group row masks must be disjoint and exhaustive")
    return masks


def load_incumbent_oof(args: argparse.Namespace) -> tuple[
    pd.DatetimeIndex, pd.Series, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]
]:
    primary_path = Path(args.primary_cache)
    residual_path = Path(args.residual_cache)
    group3_path = Path(args.group3_cache)
    candidate_path = Path(args.incumbent_candidate)
    paths = {
        "primary_cache": primary_path,
        "residual_cache": residual_path,
        "group3_cache": group3_path,
        "candidate": candidate_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hash_match = {name: hashes[name] == EXPECTED_HASHES[name] for name in paths}
    if not all(hash_match.values()):
        raise AssertionError(f"incumbent lineage SHA-256 mismatch: {hash_match}")

    baselines, truth_series, index, issues = load_frozen_validation_baselines(
        primary_path, residual_path, group3_path
    )
    truth = {name: values.to_numpy(dtype=float) for name, values in truth_series.items()}
    prediction = {name: values.to_numpy(dtype=float) for name, values in baselines.items()}
    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        residual_path, allow_pickle=False
    ) as residual:
        prediction["kpx_group_1"] = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=G1_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        prediction["kpx_group_2"] = apply_bounded_blend(
            primary["kpx_group_2__reference"],
            primary["kpx_group_2__expert"],
            weight=G2_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_2"],
        )

    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
    validate_submission(candidate, sample)
    metrics = evaluate_competition(truth, prediction)
    expected_score = 0.6491632057452089
    if abs(float(metrics["score"]) - expected_score) > 1e-12:
        raise AssertionError("incumbent OOF score did not reproduce")

    perturbed = {name: values.copy() for name, values in prediction.items()}
    perturbed[TARGET] = np.clip(perturbed[TARGET] + 50.0, 0.0, CAPACITY)
    macro_after = evaluate_competition(truth, perturbed)
    group_before = evaluate_group(truth[TARGET], prediction[TARGET], CAPACITY)
    group_after = evaluate_group(truth[TARGET], perturbed[TARGET], CAPACITY)
    group_delta = group_after.score - group_before.score
    macro_delta = float(macro_after["score"] - metrics["score"])
    if not np.isclose(macro_delta, group_delta / 3.0, atol=1e-14, rtol=0.0):
        raise AssertionError("official macro transfer is not group-3 delta / 3")

    masks = group_row_masks({name: len(values) for name, values in truth.items()})
    lineage = {
        "submission_id": 1508386,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "sha256": hashes,
        "sha256_match": hash_match,
        "candidate_rows": int(len(candidate)),
        "candidate_validated_against_sample": True,
        "oof_metrics": metrics,
        "group_row_masks": {
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "disjoint_and_exhaustive": True,
        },
        "macro_transfer_check": {
            "synthetic_group3_score_delta": float(group_delta),
            "synthetic_macro_score_delta": macro_delta,
            "expected_macro_delta": float(group_delta / 3.0),
            "passed": True,
        },
        "public_score_deltas": {
            "safe_cv_best_oof_to_public": -0.0079668315,
            "safe_cv_best_public_minus_incumbent_public": -0.0129487705,
        },
    }
    return index, issues, truth, prediction, lineage


def load_2023_data(args: argparse.Namespace) -> dict[str, Any]:
    kma_speed_series, kma_issue_series = load_context_speed(Path(args.kma_2023))
    index = pd.DatetimeIndex(kma_speed_series.index)
    labels = pd.read_csv(
        args.labels, encoding="utf-8-sig", usecols=["kst_dtm", TARGET]
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    truth = labels.set_index("kst_dtm")[TARGET].reindex(index).to_numpy(dtype=float)
    if int(np.isfinite(truth).sum()) != 8759:
        raise ValueError("unexpected 2023 group-3 label coverage")
    hub_u, hub_v, grid_x, grid_y, ldaps_issues = load_ldaps_hub_vectors(
        Path(args.ldaps_train), requested_index=index
    )
    kma_issues = kma_issue_series.reindex(index)
    availability = {
        "kma_2023": audit_availability(index, kma_issues, "kma_2023"),
        "ldaps_2023": audit_availability(index, ldaps_issues, "ldaps_2023"),
    }
    center, variants = displacement_speeds(
        hub_u, hub_v, grid_x, grid_y, ldaps_issues
    )
    return {
        "index": index,
        "truth": truth,
        "kma_speed": kma_speed_series.reindex(index).to_numpy(dtype=float),
        "kma_issues": kma_issues.to_numpy(),
        "ldaps_issues": ldaps_issues.to_numpy(),
        "center_speed": center,
        "variant_speeds": variants,
        "availability": availability,
    }


def _fold_definitions(index: pd.DatetimeIndex) -> tuple[dict[str, Any], ...]:
    return (
        {
            "fold": "2023_h1_to_q3",
            "train": np.asarray(
                (index >= pd.Timestamp("2023-01-01"))
                & (index < pd.Timestamp("2023-07-01"))
            ),
            "validation": np.asarray(
                (index >= pd.Timestamp("2023-07-01"))
                & (index < pd.Timestamp("2023-10-01"))
            ),
        },
        {
            "fold": "2023_q3_to_q4",
            "train": np.asarray(
                (index >= pd.Timestamp("2023-01-01"))
                & (index < pd.Timestamp("2023-10-01"))
            ),
            "validation": np.asarray(
                (index >= pd.Timestamp("2023-10-01"))
                & (index < pd.Timestamp("2024-01-01"))
            ),
        },
    )


def run_internal_selection(data: dict[str, Any]) -> tuple[FrozenRule | None, list[dict[str, Any]]]:
    index = data["index"]
    truth = data["truth"]
    fold_rows: list[dict[str, Any]] = []
    for fold in _fold_definitions(index):
        train = fold["train"] & np.isfinite(truth)
        validation = fold["validation"]
        anchor = fit_power_curve(
            data["kma_speed"], truth, train, data["kma_speed"]
        )
        centered = fit_power_curve(
            data["center_speed"], truth, train, data["center_speed"]
        )
        variant_predictions = {
            candidate_id: fit_power_curve(speed, truth, train, speed)
            for candidate_id, speed in data["variant_speeds"].items()
        }
        for spec in CANDIDATES:
            for weight in WEIGHTS:
                rule = FrozenRule(spec.candidate_id, weight)
                candidate = apply_overlay(
                    anchor, centered, variant_predictions[spec.candidate_id], rule
                )
                comparison = compare_group(truth, anchor, candidate, validation)
                fold_rows.append(
                    {
                        "fold": fold["fold"],
                        "candidate_id": spec.candidate_id,
                        "weight": weight,
                        "train_rows": int(train.sum()),
                        "validation_rows": int(validation.sum()),
                        **{
                            f"incumbent_{key}": comparison["incumbent"][key]
                            for key in COMPONENTS
                        },
                        **{
                            f"candidate_{key}": comparison["candidate"][key]
                            for key in COMPONENTS
                        },
                        **{
                            f"delta_{key}": comparison["delta"][key]
                            for key in COMPONENTS
                        },
                        "changed_rows": comparison["changed_rows"],
                    }
                )

    frame = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    for (candidate_id, weight), rows in frame.groupby(["candidate_id", "weight"], sort=False):
        stable = bool(
            (rows["delta_score"] > 0.0).all()
            and (rows["delta_ficr"] > 0.0).all()
            and (rows["delta_one_minus_nmae"] >= -0.002).all()
        )
        summaries.append(
            {
                "candidate_id": str(candidate_id),
                "weight": float(weight),
                "stable": stable,
                "mean_score_delta": float(rows["delta_score"].mean()),
                "worst_score_delta": float(rows["delta_score"].min()),
                "mean_ficr_delta": float(rows["delta_ficr"].mean()),
                "worst_nmae_delta": float(rows["delta_one_minus_nmae"].min()),
            }
        )
    stable = [row for row in summaries if row["stable"]]
    if not stable:
        return None, fold_rows
    selected = max(
        stable,
        key=lambda row: (
            row["mean_score_delta"],
            row["worst_score_delta"],
            -row["weight"],
            row["candidate_id"],
        ),
    )
    return FrozenRule(selected["candidate_id"], selected["weight"]), fold_rows


def target_date_block_bootstrap(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    dates = target_dates(index)
    unique_dates = pd.DatetimeIndex(dates.unique())
    positions = [np.flatnonzero(dates == date) for date in unique_dates]
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(positions), size=len(positions))
        rows = np.concatenate([positions[position] for position in sampled])
        before = evaluate_group(truth[rows], incumbent[rows], CAPACITY)
        after = evaluate_group(truth[rows], candidate[rows], CAPACITY)
        deltas[repetition] = after.score - before.score
    return {
        "repetitions": int(repetitions),
        "blocks": int(len(unique_dates)),
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "q05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "mean": float(np.mean(deltas)),
        "q95": float(np.quantile(deltas, 0.95)),
    }


def _diagnostics(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    issues: np.ndarray,
) -> dict[str, Any]:
    h2_start = pd.Timestamp("2024-07-01 01:00:00")
    periods = {
        "full": np.ones(len(index), dtype=bool),
        "h1": np.asarray(index < h2_start),
        "h2": np.asarray(index >= h2_start),
    }
    result: dict[str, Any] = {
        "periods": {
            name: compare_group(truth, incumbent, candidate, mask)
            for name, mask in periods.items()
        }
    }
    result["months"] = {}
    for month in range(1, 13):
        mask = np.asarray((index.year == 2024) & (index.month == month))
        result["months"][str(month)] = compare_group(
            truth, incumbent, candidate, mask
        )
    seasons = {
        "DJF": (12, 1, 2),
        "MAM": (3, 4, 5),
        "JJA": (6, 7, 8),
        "SON": (9, 10, 11),
    }
    result["seasons"] = {
        name: compare_group(
            truth,
            incumbent,
            candidate,
            np.asarray((index.year == 2024) & index.month.isin(months)),
        )
        for name, months in seasons.items()
    }
    issue_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(issues)))
    lead = (index - issue_index).total_seconds().to_numpy(dtype=float) / 3600.0
    lead_masks = {
        "12-17": (lead >= 12.0) & (lead < 18.0),
        "18-23": (lead >= 18.0) & (lead < 24.0),
        "24-29": (lead >= 24.0) & (lead < 30.0),
        "30-35": (lead >= 30.0) & (lead <= 35.0),
    }
    result["lead_time_hours"] = {
        name: compare_group(truth, incumbent, candidate, mask)
        for name, mask in lead_masks.items()
        if int(mask.sum()) >= 24
    }
    return result


def _load_2024_displacement_inputs(
    args: argparse.Namespace,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    hub_u, hub_v, grid_x, grid_y, issues = load_ldaps_hub_vectors(
        Path(args.ldaps_train), requested_index=index
    )
    availability = audit_availability(index, issues, "ldaps_2024")
    center, variants = displacement_speeds(
        hub_u, hub_v, grid_x, grid_y, issues
    )
    return {
        "issues": issues.to_numpy(),
        "center_speed": center,
        "variant_speeds": variants,
        "availability": availability,
    }


def _promotion_gates(
    diagnostics: dict[str, Any],
    bootstrap: dict[str, float | int],
    group12_identical: bool,
) -> dict[str, bool]:
    full = diagnostics["periods"]["full"]["delta"]
    h1 = diagnostics["periods"]["h1"]["delta"]
    h2 = diagnostics["periods"]["h2"]["delta"]
    season_score = [row["delta"]["score"] for row in diagnostics["seasons"].values()]
    return {
        "g3_score_delta_at_least_0p0045": full["score"] >= 0.0045,
        "h1_score_delta_positive": h1["score"] > 0.0,
        "h2_score_delta_positive": h2["score"] > 0.0,
        "bootstrap_probability_at_least_0p80": bootstrap["p_delta_gt_0"] >= 0.80,
        "bootstrap_q05_at_least_minus_0p002": bootstrap["q05"] >= -0.002,
        "g3_nmae_delta_at_least_minus_0p002": full["one_minus_nmae"] >= -0.002,
        "g3_ficr_delta_positive": full["ficr"] > 0.0,
        "no_season_below_minus_0p005": min(season_score) >= -0.005,
        "group1_group2_exactly_identical": group12_identical,
    }


def _write_candidate_if_promoted(
    args: argparse.Namespace,
    rule: FrozenRule,
    train_data: dict[str, Any],
) -> dict[str, Any]:
    incumbent_path = Path(args.incumbent_candidate)
    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    incumbent = pd.read_csv(incumbent_path, encoding="utf-8-sig")
    validate_submission(incumbent, sample)
    index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]))
    hub_u, hub_v, grid_x, grid_y, issues = load_ldaps_hub_vectors(
        Path(args.ldaps_test), requested_index=index
    )
    test_availability = audit_availability(index, issues, "ldaps_test")
    test_center, test_variants = displacement_speeds(
        hub_u, hub_v, grid_x, grid_y, issues
    )

    labels = pd.read_csv(
        args.labels, encoding="utf-8-sig", usecols=["kst_dtm", TARGET]
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    train_index = pd.DatetimeIndex(
        labels.loc[labels[TARGET].notna(), "kst_dtm"]
    )
    train_truth = labels.loc[labels[TARGET].notna(), TARGET].to_numpy(dtype=float)
    train_u, train_v, train_x, train_y, train_issues = load_ldaps_hub_vectors(
        Path(args.ldaps_train), requested_index=train_index
    )
    audit_availability(train_index, train_issues, "ldaps_full_train")
    train_center, train_variants = displacement_speeds(
        train_u, train_v, train_x, train_y, train_issues
    )
    train_mask = np.ones(len(train_truth), dtype=bool)
    center_member = fit_power_curve(
        train_center, train_truth, train_mask, test_center
    )
    displaced_member = fit_power_curve(
        train_variants[rule.candidate_id],
        train_truth,
        train_mask,
        test_variants[rule.candidate_id],
    )
    output = incumbent.copy()
    output[TARGET] = apply_overlay(
        incumbent[TARGET].to_numpy(dtype=float), center_member, displaced_member, rule
    )
    if not np.array_equal(
        output["kpx_group_1"].to_numpy(), incumbent["kpx_group_1"].to_numpy()
    ) or not np.array_equal(
        output["kpx_group_2"].to_numpy(), incumbent["kpx_group_2"].to_numpy()
    ):
        raise AssertionError("candidate changed incumbent group 1 or group 2")
    validate_submission(output, sample)
    output_path = Path(args.output_candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    reloaded = pd.read_csv(output_path, encoding="utf-8")
    validate_submission(reloaded, sample)
    changed = {
        group: int(
            np.sum(
                reloaded[group].to_numpy(dtype=float)
                != incumbent[group].to_numpy(dtype=float)
            )
        )
        for group in CAPACITY_KWH
    }
    if changed["kpx_group_1"] or changed["kpx_group_2"]:
        raise AssertionError("serialized candidate changed incumbent group 1 or group 2")
    return {
        "created": True,
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "rows": int(len(reloaded)),
        "changed_rows": changed,
        "test_availability": test_availability,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_2023_data(args)
    fold = _fold_definitions(data["index"])[0]
    train = fold["train"] & np.isfinite(data["truth"])
    anchor = fit_power_curve(
        data["kma_speed"], data["truth"], train, data["kma_speed"]
    )
    center = fit_power_curve(
        data["center_speed"], data["truth"], train, data["center_speed"]
    )
    spec = CANDIDATES[0]
    member = fit_power_curve(
        data["variant_speeds"][spec.candidate_id],
        data["truth"],
        train,
        data["variant_speeds"][spec.candidate_id],
    )
    rule = FrozenRule(spec.candidate_id, WEIGHTS[0])
    candidate = apply_overlay(anchor, center, member, rule)
    report = {
        "smoke_passed": True,
        "opened_2024_candidate_evaluation": False,
        "candidate": asdict(rule),
        "fold": fold["fold"],
        "comparison": compare_group(
            data["truth"], anchor, candidate, fold["validation"]
        ),
        "availability": data["availability"],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _json_dump(Path(args.artifact_dir) / "smoke_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    report_path = artifact_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{report_path} already exists; use --overwrite only for an intentional rerun"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "written_before_candidate_execution": True,
        "selection_period": "2023 only",
        "folds": [fold["fold"] for fold in _fold_definitions(pd.date_range("2023-01-01", periods=1))],
        "candidates": [asdict(spec) for spec in CANDIDATES],
        "weights": list(WEIGHTS),
        "movement_cap_ratio": MAXIMUM_MOVEMENT_RATIO,
        "post_2024_retuning_allowed": False,
    }
    _json_dump(artifact_dir / "preregistered_candidates.json", preregistration)

    incumbent_index, incumbent_issues, truth_all, prediction_all, lineage = load_incumbent_oof(args)
    train_data = load_2023_data(args)
    selected_rule, internal_rows = run_internal_selection(train_data)
    internal_frame = pd.DataFrame(internal_rows)
    internal_frame.to_csv(artifact_dir / "internal_2023_results.csv", index=False, encoding="utf-8")
    internal_summary = (
        internal_frame.groupby(["candidate_id", "weight"], as_index=False)
        .agg(
            mean_score_delta=("delta_score", "mean"),
            worst_score_delta=("delta_score", "min"),
            mean_nmae_delta=("delta_one_minus_nmae", "mean"),
            worst_nmae_delta=("delta_one_minus_nmae", "min"),
            mean_ficr_delta=("delta_ficr", "mean"),
            worst_ficr_delta=("delta_ficr", "min"),
        )
    )
    internal_summary["stable"] = (
        (internal_summary["worst_score_delta"] > 0.0)
        & (internal_summary["worst_ficr_delta"] > 0.0)
        & (internal_summary["worst_nmae_delta"] >= -0.002)
    )
    internal_summary.to_csv(
        artifact_dir / "internal_2023_summary.csv", index=False, encoding="utf-8"
    )

    report: dict[str, Any] = {
        "experiment_id": "s066_g3_frozen_displacement_20260805",
        "hypothesis": "an issue-local LDAPS displacement increment improves only incumbent group 3",
        "lineage": lineage,
        "preregistration": preregistration,
        "availability_2023": train_data["availability"],
        "internal_selection": {
            "selected_rule": None if selected_rule is None else asdict(selected_rule),
            "stable_rule_count": int(internal_summary["stable"].sum()),
            "results_file": (artifact_dir / "internal_2023_results.csv").as_posix(),
            "summary_file": (artifact_dir / "internal_2023_summary.csv").as_posix(),
        },
        "warning": "2024 has been observed by earlier experiments and is not described as untouched.",
    }
    if selected_rule is None:
        report.update(
            {
                "stopped_before_2024_candidate_evaluation": True,
                "promotion_passed": False,
                "candidate_csv": {"created": False},
                "stop_reason": "no candidate was positive in Score and FICR on both 2023 forward folds",
                "runtime_seconds": float(time.perf_counter() - started),
            }
        )
        _json_dump(report_path, report)
        return report

    full_2023_train = np.isfinite(train_data["truth"])
    centered_2024_inputs = _load_2024_displacement_inputs(args, incumbent_index)
    centered_member_2024 = fit_power_curve(
        train_data["center_speed"],
        train_data["truth"],
        full_2023_train,
        centered_2024_inputs["center_speed"],
    )
    displaced_member_2024 = fit_power_curve(
        train_data["variant_speeds"][selected_rule.candidate_id],
        train_data["truth"],
        full_2023_train,
        centered_2024_inputs["variant_speeds"][selected_rule.candidate_id],
    )
    incumbent_g3 = prediction_all[TARGET]
    candidate_g3 = apply_overlay(
        incumbent_g3, centered_member_2024, displaced_member_2024, selected_rule
    )
    candidate_all = {name: values.copy() for name, values in prediction_all.items()}
    candidate_all[TARGET] = candidate_g3
    group12_identical = bool(
        np.array_equal(candidate_all["kpx_group_1"], prediction_all["kpx_group_1"])
        and np.array_equal(candidate_all["kpx_group_2"], prediction_all["kpx_group_2"])
    )
    diagnostics = _diagnostics(
        truth_all[TARGET],
        incumbent_g3,
        candidate_g3,
        incumbent_index,
        centered_2024_inputs["issues"],
    )
    bootstrap = target_date_block_bootstrap(
        truth_all[TARGET],
        incumbent_g3,
        candidate_g3,
        incumbent_index,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    gates = _promotion_gates(diagnostics, bootstrap, group12_identical)
    promotion_passed = bool(all(gates.values()))
    incumbent_macro = evaluate_competition(truth_all, prediction_all)
    candidate_macro = evaluate_competition(truth_all, candidate_all)
    macro_delta = {
        key: float(candidate_macro[key] - incumbent_macro[key])
        for key in COMPONENTS
    }
    g3_delta = diagnostics["periods"]["full"]["delta"]
    for key in COMPONENTS:
        if not np.isclose(macro_delta[key], g3_delta[key] / 3.0, atol=1e-13, rtol=0.0):
            raise AssertionError(f"macro delta transfer failed for {key}")
    np.savez_compressed(
        artifact_dir / "evaluation_2024_predictions.npz",
        index_ns=incumbent_index.astype("int64").to_numpy(),
        issue_ns=pd.to_datetime(centered_2024_inputs["issues"]).astype("int64"),
        truth=truth_all[TARGET].astype("float32"),
        incumbent=incumbent_g3.astype("float32"),
        candidate=candidate_g3.astype("float32"),
        centered_member=centered_member_2024.astype("float32"),
        displaced_member=displaced_member_2024.astype("float32"),
    )
    candidate_record: dict[str, Any] = {"created": False}
    if promotion_passed:
        candidate_record = _write_candidate_if_promoted(args, selected_rule, train_data)

    report.update(
        {
            "stopped_before_2024_candidate_evaluation": False,
            "frozen_rule": asdict(selected_rule),
            "availability_2024": centered_2024_inputs["availability"],
            "evaluation_2024": {
                "diagnostics": diagnostics,
                "bootstrap": bootstrap,
                "incumbent_macro": incumbent_macro,
                "candidate_macro": candidate_macro,
                "macro_delta": macro_delta,
                "group1_group2_identical": group12_identical,
                "evaluated_once_after_freeze": True,
            },
            "promotion_gates": gates,
            "promotion_passed": promotion_passed,
            "candidate_csv": candidate_record,
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    _json_dump(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/group3_frozen_displacement")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--ldaps-train", default="data/train/ldaps_train.csv")
    parser.add_argument("--ldaps-test", default="data/test/ldaps_test.csv")
    parser.add_argument(
        "--kma-2023",
        default="artifacts_final/external_weather/kma_um_regional_context_2023/features.csv",
    )
    parser.add_argument(
        "--primary-cache",
        default="artifacts_final/lineage/kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz",
    )
    parser.add_argument(
        "--residual-cache",
        default="artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz",
    )
    parser.add_argument(
        "--group3-cache",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof_20260725.npz",
    )
    parser.add_argument(
        "--incumbent-candidate",
        default="artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv",
    )
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument(
        "--output-candidate",
        default="submissions/group3_frozen_displacement_20260805.csv",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = smoke(args) if args.smoke else run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
