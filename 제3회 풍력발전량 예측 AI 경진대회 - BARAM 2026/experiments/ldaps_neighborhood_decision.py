from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.kma_um_power_curve_gate import (
    CAPACITY,
    COMPONENTS,
    TARGET,
    _compare,
    issue_block_bootstrap,
)
from src.metrics import CAPACITY_KWH


Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
KEYS = ("forecast_kst_dtm", "data_available_kst_dtm")
LDAPS_COLUMNS = (
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "grid_id",
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
)
SCENARIO_OFFSETS = (-0.08, -0.06, 0.0, 0.06, 0.08)


@dataclass(frozen=True)
class DecisionPolicy:
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_disagreement_kwh: float
    spread_mode: str
    minimum_spread_kwh: float | None
    maximum_spread_kwh: float | None
    alpha: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ldaps_hub_scenarios(
    path: Path,
    requested_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(LDAPS_COLUMNS),
    )
    for key in KEYS:
        frame[key] = pd.to_datetime(frame[key])
    if frame.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise ValueError("LDAPS contains duplicate forecast/grid rows")

    issue_counts = frame.groupby("forecast_kst_dtm")[
        "data_available_kst_dtm"
    ].nunique()
    if int(issue_counts.max()) != 1:
        raise ValueError("LDAPS contains multiple issue cycles for one target")

    u10 = pd.to_numeric(
        frame["heightAboveGround_10_10u"], errors="coerce"
    ).to_numpy(dtype=float)
    v10 = pd.to_numeric(
        frame["heightAboveGround_10_10v"], errors="coerce"
    ).to_numpy(dtype=float)
    u50 = (
        pd.to_numeric(
            frame["heightAboveGround_50_50MUmax"], errors="coerce"
        ).to_numpy(dtype=float)
        + pd.to_numeric(
            frame["heightAboveGround_50_50MUmin"], errors="coerce"
        ).to_numpy(dtype=float)
    ) / 2.0
    v50 = (
        pd.to_numeric(
            frame["heightAboveGround_50_50MVmax"], errors="coerce"
        ).to_numpy(dtype=float)
        + pd.to_numeric(
            frame["heightAboveGround_50_50MVmin"], errors="coerce"
        ).to_numpy(dtype=float)
    ) / 2.0
    lower_speed = np.clip(np.hypot(u10, v10), 0.05, None)
    upper_speed = np.clip(np.hypot(u50, v50), 0.05, None)
    alpha = np.log(upper_speed / lower_speed) / np.log(5.0)
    alpha = np.nan_to_num(alpha, nan=0.14, posinf=0.14, neginf=0.14)
    alpha = np.clip(alpha, -0.30, 0.60)
    hub_speed = np.clip(upper_speed * (117.0 / 50.0) ** alpha, 0.0, 45.0)
    if not np.isfinite(hub_speed).all():
        raise ValueError("LDAPS hub-height wind contains non-finite values")

    local = frame.loc[:, ["forecast_kst_dtm", "grid_id"]].copy()
    local["hub_ws117"] = hub_speed
    scenarios = local.pivot(
        index="forecast_kst_dtm",
        columns="grid_id",
        values="hub_ws117",
    ).sort_index()
    scenarios.columns = [int(value) for value in scenarios.columns]
    issues = (
        frame.drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")["data_available_kst_dtm"]
        .sort_index()
    )

    if requested_index is not None:
        scenarios = scenarios.reindex(requested_index)
        issues = issues.reindex(requested_index)
    if scenarios.shape[1] != 16:
        raise ValueError(
            f"expected 16 LDAPS neighborhood grids, found {scenarios.shape[1]}"
        )
    if scenarios.isna().any().any() or not np.isfinite(
        scenarios.to_numpy(dtype=float)
    ).all():
        raise ValueError("LDAPS neighborhood coverage is incomplete")
    if issues.isna().any():
        raise ValueError("LDAPS issue-time coverage is incomplete")
    scenarios.index.name = "forecast_kst_dtm"
    return scenarios, issues


def fit_neighborhood_power_curve(
    train_wind: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    predict_wind: np.ndarray,
) -> np.ndarray:
    train_wind = np.asarray(train_wind, dtype=float)
    truth = np.asarray(truth, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    predict_wind = np.asarray(predict_wind, dtype=float)
    if train_wind.ndim != 2 or predict_wind.ndim != 2:
        raise ValueError("neighborhood wind arrays must be two-dimensional")
    if train_wind.shape[0] != len(truth) or len(train_mask) != len(truth):
        raise ValueError("power-curve training arrays are misaligned")
    if int(train_mask.sum()) < 1_000:
        raise ValueError("neighborhood power curve needs 1,000 timestamps")
    x = train_wind[train_mask].reshape(-1)
    y = np.repeat(truth[train_mask], train_wind.shape[1])
    valid = np.isfinite(x) & np.isfinite(y)
    model = IsotonicRegression(
        y_min=0.0,
        y_max=CAPACITY,
        out_of_bounds="clip",
    )
    model.fit(x[valid], y[valid])
    prediction = np.asarray(
        model.predict(predict_wind.reshape(-1)), dtype=float
    ).reshape(predict_wind.shape)
    if not np.isfinite(prediction).all():
        raise ValueError("neighborhood power curve produced non-finite values")
    return prediction


def center_neighborhood_scenarios(
    power_scenarios: np.ndarray,
    incumbent: np.ndarray,
) -> np.ndarray:
    """Bias-correct the ensemble center while preserving spatial anomalies."""
    power_scenarios = np.asarray(power_scenarios, dtype=float)
    incumbent = np.asarray(incumbent, dtype=float)
    if power_scenarios.ndim != 2:
        raise ValueError("power scenarios must be two-dimensional")
    if power_scenarios.shape[0] != len(incumbent):
        raise ValueError("scenario and incumbent rows are misaligned")
    center = np.median(power_scenarios, axis=1)
    centered = incumbent[:, None] + power_scenarios - center[:, None]
    return np.clip(centered, 0.0, CAPACITY)


def _scenario_score(samples: np.ndarray, action: float) -> float:
    eligible = samples >= 0.10 * CAPACITY
    if not eligible.any():
        return float("nan")
    actual = samples[eligible]
    error_rate = np.abs(actual - action) / CAPACITY
    unit_fraction = np.where(
        error_rate <= 0.06,
        1.0,
        np.where(error_rate <= 0.08, 0.75, 0.0),
    )
    nmae = float(error_rate.mean())
    ficr = float(np.sum(actual * unit_fraction) / np.sum(actual))
    return 0.5 * (1.0 - nmae) + 0.5 * ficr


def optimize_neighborhood_actions(
    power_scenarios: np.ndarray,
    incumbent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power_scenarios = np.asarray(power_scenarios, dtype=float)
    incumbent = np.asarray(incumbent, dtype=float)
    if power_scenarios.ndim != 2:
        raise ValueError("power scenarios must be two-dimensional")
    if power_scenarios.shape[0] != len(incumbent):
        raise ValueError("scenario and incumbent rows are misaligned")

    actions = incumbent.copy()
    incumbent_scores = np.full(len(incumbent), np.nan, dtype=float)
    optimized_scores = np.full(len(incumbent), np.nan, dtype=float)
    offsets = np.asarray(SCENARIO_OFFSETS, dtype=float) * CAPACITY
    for row in range(len(incumbent)):
        samples = power_scenarios[row]
        base_score = _scenario_score(samples, incumbent[row])
        incumbent_scores[row] = base_score
        if not np.isfinite(base_score):
            optimized_scores[row] = base_score
            continue
        candidates = np.unique(
            np.clip(
                np.concatenate(
                    ([incumbent[row]], (samples[:, None] + offsets).reshape(-1))
                ),
                0.0,
                CAPACITY,
            )
        )
        scores = np.asarray(
            [_scenario_score(samples, candidate) for candidate in candidates],
            dtype=float,
        )
        best_score = float(np.nanmax(scores))
        tied = np.flatnonzero(np.isclose(scores, best_score, atol=1e-12))
        best = tied[
            np.argmin(np.abs(candidates[tied] - incumbent[row]))
        ]
        actions[row] = candidates[best]
        optimized_scores[row] = scores[best]
    return actions, incumbent_scores, optimized_scores


def decision_gate(
    action: np.ndarray,
    incumbent: np.ndarray,
    spread_kwh: np.ndarray,
    expected_gain: np.ndarray,
    available: np.ndarray,
    policy: DecisionPolicy,
) -> np.ndarray:
    difference = action - incumbent
    if policy.direction == "up":
        direction = difference > 0.0
    elif policy.direction == "down":
        direction = difference < 0.0
    elif policy.direction == "both":
        direction = difference != 0.0
    else:
        raise ValueError(f"unknown direction: {policy.direction}")
    if policy.spread_mode not in {"all", "low", "high"}:
        raise ValueError(f"unknown spread mode: {policy.spread_mode}")
    spread = np.ones(len(incumbent), dtype=bool)
    if policy.minimum_spread_kwh is not None:
        spread &= spread_kwh >= policy.minimum_spread_kwh
    if policy.maximum_spread_kwh is not None:
        spread &= spread_kwh <= policy.maximum_spread_kwh
    ratio = incumbent / CAPACITY
    return (
        available
        & np.isfinite(expected_gain)
        & (expected_gain > 1e-12)
        & direction
        & (np.abs(difference) >= policy.minimum_disagreement_kwh)
        & spread
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )


def apply_decision_policy(
    action: np.ndarray,
    incumbent: np.ndarray,
    spread_kwh: np.ndarray,
    expected_gain: np.ndarray,
    available: np.ndarray,
    policy: DecisionPolicy,
    *,
    maximum_incremental_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < maximum_incremental_movement_ratio <= 0.01:
        raise ValueError("incremental movement ratio must be in (0, 0.01]")
    gate = decision_gate(
        action,
        incumbent,
        spread_kwh,
        expected_gain,
        available,
        policy,
    )
    bound = maximum_incremental_movement_ratio * CAPACITY
    movement = np.clip(policy.alpha * (action - incumbent), -bound, bound)
    candidate = incumbent.copy()
    candidate[gate] = incumbent[gate] + movement[gate]
    return np.clip(candidate, 0.0, CAPACITY), gate


def select_decision_policy(
    truth: np.ndarray,
    incumbent: np.ndarray,
    action: np.ndarray,
    spread_kwh: np.ndarray,
    expected_gain: np.ndarray,
    available: np.ndarray,
    selection: np.ndarray,
    *,
    minimum_changed_rows: int,
    maximum_changed_ratio: float,
    maximum_incremental_movement_ratio: float,
) -> dict[str, object]:
    difference = action - incumbent
    positive_rows: list[dict[str, object]] = []
    screened_rows: list[dict[str, object]] = []
    base_pool = selection & available & np.isfinite(expected_gain)
    if not base_pool.any():
        return {
            "selected": None,
            "top_positive": [],
            "top_screened": [],
            "screened_policies": 0,
            "positive_policies": 0,
        }
    spread_cutoff = float(np.median(spread_kwh[base_pool]))
    spread_specs = (
        ("all", None, None),
        ("low", None, spread_cutoff),
        ("high", spread_cutoff, None),
    )
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = difference > 0.0
        elif direction_name == "down":
            direction = difference < 0.0
        else:
            direction = difference != 0.0
        pool_mask = (
            base_pool
            & direction
            & (expected_gain > 1e-12)
            & (np.abs(difference) > 1e-12)
        )
        if not pool_mask.any():
            continue
        pool = np.abs(difference[pool_mask])
        for coverage in (0.10, 0.05):
            threshold = float(np.quantile(pool, 1.0 - coverage))
            for spread_mode, minimum_spread, maximum_spread in spread_specs:
                for minimum_ratio, maximum_ratio in (
                    (0.10, 0.80),
                    (0.20, 0.80),
                ):
                    for alpha in (0.10, 0.25, 0.50, 1.00):
                        policy = DecisionPolicy(
                            direction=direction_name,
                            coverage=coverage,
                            minimum_base_ratio=minimum_ratio,
                            maximum_base_ratio=maximum_ratio,
                            minimum_disagreement_kwh=threshold,
                            spread_mode=spread_mode,
                            minimum_spread_kwh=minimum_spread,
                            maximum_spread_kwh=maximum_spread,
                            alpha=alpha,
                        )
                        candidate, gate = apply_decision_policy(
                            action,
                            incumbent,
                            spread_kwh,
                            expected_gain,
                            available,
                            policy,
                            maximum_incremental_movement_ratio=(
                                maximum_incremental_movement_ratio
                            ),
                        )
                        changed = selection & gate & (
                            np.abs(candidate - incumbent) > 1e-9
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
                        record = {
                            "policy": policy,
                            "comparison": comparison,
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
                        screened_rows.append(record)
                        if (
                            min(
                                comparison["delta"][component]
                                for component in COMPONENTS
                            )
                            > 0.0
                        ):
                            positive_rows.append(record)

    def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
        delta = row["comparison"]["delta"]
        return (
            float(delta["score"]),
            float(min(delta["one_minus_nmae"], delta["ficr"])),
            -float(row["mean_absolute_increment_kwh"]),
        )

    screened_rows.sort(key=sort_key, reverse=True)
    positive_rows.sort(key=sort_key, reverse=True)

    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {**row, "policy": row["policy"].to_dict()}

    return {
        "selected": None if not positive_rows else positive_rows[0],
        "top_positive": [serialize(row) for row in positive_rows[:10]],
        "top_screened": [serialize(row) for row in screened_rows[:10]],
        "screened_policies": len(screened_rows),
        "positive_policies": len(positive_rows),
    }


def _month_deltas(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    h2: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        mask = h2 & np.asarray(index.month == month)
        if int(mask.sum()) >= 24:
            output[str(month)] = _compare(
                truth, incumbent, candidate, mask
            )["delta"]
    return output


def _seasonal_deltas(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    h2: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "july_to_september": _compare(
            truth,
            incumbent,
            candidate,
            h2 & np.asarray(index.month <= 9),
        )["delta"],
        "october_to_december": _compare(
            truth,
            incumbent,
            candidate,
            h2 & np.asarray(index.month >= 10),
        )["delta"],
    }


def _policy_from_dict(raw: dict[str, object]) -> DecisionPolicy:
    return DecisionPolicy(
        direction=str(raw["direction"]),
        coverage=float(raw["coverage"]),
        minimum_base_ratio=float(raw["minimum_base_ratio"]),
        maximum_base_ratio=float(raw["maximum_base_ratio"]),
        minimum_disagreement_kwh=float(raw["minimum_disagreement_kwh"]),
        spread_mode=str(raw["spread_mode"]),
        minimum_spread_kwh=(
            None
            if raw.get("minimum_spread_kwh") is None
            else float(raw["minimum_spread_kwh"])
        ),
        maximum_spread_kwh=(
            None
            if raw.get("maximum_spread_kwh") is None
            else float(raw["maximum_spread_kwh"])
        ),
        alpha=float(raw["alpha"]),
    )


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_submission(
    *,
    train_wind: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    policy: DecisionPolicy,
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
    test_wind_frame, _ = load_ldaps_hub_scenarios(test_ldaps, index)
    raw_test_power = fit_neighborhood_power_curve(
        train_wind,
        truth,
        train_mask,
        test_wind_frame.to_numpy(dtype=float),
    )
    incumbent = incumbent_frame[TARGET].to_numpy(dtype=float)
    test_power = center_neighborhood_scenarios(
        raw_test_power, incumbent
    )
    action, base_score, optimized_score = optimize_neighborhood_actions(
        test_power, incumbent
    )
    spread = np.std(test_power, axis=1)
    expected_gain = optimized_score - base_score
    candidate, gate = apply_decision_policy(
        action,
        incumbent,
        spread,
        expected_gain,
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
    config = load_config(Path.cwd())
    audit = CandidateValidator(config).audit(output_submission)
    if not audit.valid:
        raise RuntimeError(f"candidate validation failed: {audit.errors}")
    changed = np.abs(candidate - incumbent) > 1e-9
    if not output[
        ["kpx_group_1", "kpx_group_2"]
    ].equals(incumbent_frame[["kpx_group_1", "kpx_group_2"]]):
        raise RuntimeError("group-1 or group-2 changed unexpectedly")
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
    issues = pd.to_datetime(cache["issue_ns"]).to_numpy()
    static_incumbent = cache["static_candidate"].astype(float)
    rolling_incumbent = cache["rolling_candidate"].astype(float)
    available = cache["available"].astype(bool)
    q1 = (index < Q2_START) & available
    q2 = (index >= Q2_START) & (index < H2_START) & available
    h1 = (index < H2_START) & available
    h2 = (index >= H2_START) & (index < END) & available
    if not np.array_equal(q2, cache["q2"].astype(bool)):
        raise ValueError("Q2 split does not match the locked power-curve cache")
    if not np.array_equal(h2, cache["h2"].astype(bool)):
        raise ValueError("H2 split does not match the locked power-curve cache")

    wind_frame, ldaps_issues = load_ldaps_hub_scenarios(
        Path(args.train_ldaps), index
    )
    wind = wind_frame.to_numpy(dtype=float)
    issue_error_hours = np.max(
        np.abs(
            (
                pd.to_datetime(ldaps_issues.to_numpy())
                - pd.to_datetime(issues)
            )
            / np.timedelta64(1, "h")
        )
    )
    if float(issue_error_hours) > 0.0:
        raise ValueError("LDAPS issue times do not match the locked OOF cache")

    q1_raw_power = fit_neighborhood_power_curve(wind, truth, q1, wind)
    q1_power = center_neighborhood_scenarios(
        q1_raw_power, static_incumbent
    )
    q1_action, q1_base_score, q1_optimized_score = (
        optimize_neighborhood_actions(q1_power, static_incumbent)
    )
    q1_spread = np.std(q1_power, axis=1)
    q1_expected_gain = q1_optimized_score - q1_base_score
    selection = select_decision_policy(
        truth,
        static_incumbent,
        q1_action,
        q1_spread,
        q1_expected_gain,
        available,
        q2,
        minimum_changed_rows=args.minimum_selection_rows,
        maximum_changed_ratio=args.maximum_changed_ratio,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    report: dict[str, Any] = {
        "family": "ldaps_neighborhood_decision",
        "method": {
            "spatial_scenarios": (
                "16 LDAPS grids transformed through one pooled monotone "
                "hub-wind-to-power curve"
            ),
            "scenario_centering": (
                "median scenario is bias-corrected to the incumbent; "
                "unscaled spatial anomalies are preserved"
            ),
            "action_objective": (
                "maximize expected official group score over spatial scenarios"
            ),
            "scenario_offsets_capacity_ratio": list(SCENARIO_OFFSETS),
        },
        "contract": {
            "q1_model_train": "2024-01-01 through 2024-03-31",
            "q2_policy_selection": (
                "2024-04-01 through 2024-07-01 00:00"
            ),
            "locked_h2": (
                "H1-refit; 2024-07-01 01:00 through 2024-12-31 23:00"
            ),
            "locked_subperiods": [
                "July through September",
                "October through December",
                "each calendar month",
                "issue-cycle block bootstrap",
            ],
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
            "grids": wind.shape[1],
            "q1_rows": int(q1.sum()),
            "q2_rows": int(q2.sum()),
            "h1_rows": int(h1.sum()),
            "h2_rows": int(h2.sum()),
            "issue_time_max_error_hours": float(issue_error_hours),
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
            "q2_all_components_positive": False
        }
        report["decision"] = {
            "tier": "rejected",
            "submission_requested": bool(
                args.write_submission_if_qualified
            ),
            "submission_created": False,
            "submission_eligible": False,
            "reason": (
                "no sparse neighborhood-decision policy improved every "
                "component on Q2"
            ),
        }
        report["submission"] = None
        _write_json(Path(args.output), report)
        return report

    policy = selection["selected"]["policy"]
    static_candidate, static_gate = apply_decision_policy(
        q1_action,
        static_incumbent,
        q1_spread,
        q1_expected_gain,
        available,
        policy,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    static_locked = _compare(
        truth, static_incumbent, static_candidate, h2
    )

    h1_raw_power = fit_neighborhood_power_curve(wind, truth, h1, wind)
    h1_power = center_neighborhood_scenarios(
        h1_raw_power, rolling_incumbent
    )
    h1_action, h1_base_score, h1_optimized_score = (
        optimize_neighborhood_actions(h1_power, rolling_incumbent)
    )
    h1_spread = np.std(h1_power, axis=1)
    h1_expected_gain = h1_optimized_score - h1_base_score
    locked_candidate, locked_gate = apply_decision_policy(
        h1_action,
        rolling_incumbent,
        h1_spread,
        h1_expected_gain,
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
        "q2_all_components_positive": min(
            selection["selected"]["comparison"]["delta"][component]
            for component in COMPONENTS
        )
        > 0.0,
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
            train_wind=wind,
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
            "neighborhood-decision candidate passed every locked gate"
            if qualified
            else "neighborhood-decision candidate failed at least one locked gate"
        ),
    }
    report["submission"] = submission
    _write_json(Path(args.output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ldaps",
        default="data/train/ldaps_train.csv",
    )
    parser.add_argument(
        "--test-ldaps",
        default="data/test/ldaps_test.csv",
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
            "ldaps_neighborhood_decision_20260726.json"
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
            "ldaps_neighborhood_decision_20260726.csv"
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
