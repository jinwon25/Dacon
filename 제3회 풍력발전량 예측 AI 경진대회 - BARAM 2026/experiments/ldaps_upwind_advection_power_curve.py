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
from experiments.ldaps_neighborhood_decision import (
    _month_deltas,
    _seasonal_deltas,
)
from src.features import TURBINES_BY_GROUP


Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
UPWIND_DISTANCES_KM = (0.0, 1.5, 3.0, 4.5)
KERNEL_SIGMA_KM = 1.5
LDAPS_COLUMNS = (
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "grid_id",
    "latitude",
    "longitude",
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
)


@dataclass(frozen=True)
class AdvectionPolicy:
    upwind_distance_km: float
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_disagreement_kwh: float
    alpha: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_ldaps_hub_vectors(
    path: Path,
    requested_index: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(LDAPS_COLUMNS),
    )
    frame["forecast_kst_dtm"] = pd.to_datetime(
        frame["forecast_kst_dtm"]
    )
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    if frame.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise ValueError("LDAPS contains duplicate forecast/grid rows")
    issue_counts = frame.groupby("forecast_kst_dtm")[
        "data_available_kst_dtm"
    ].nunique()
    if int(issue_counts.max()) != 1:
        raise ValueError("LDAPS contains multiple issues for one target")

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
    shear = np.log(upper_speed / lower_speed) / np.log(5.0)
    shear = np.nan_to_num(
        shear, nan=0.14, posinf=0.14, neginf=0.14
    )
    shear = np.clip(shear, -0.30, 0.60)
    hub_speed = np.clip(
        upper_speed * (117.0 / 50.0) ** shear, 0.0, 45.0
    )
    ratio = hub_speed / upper_speed
    frame = frame.loc[
        :,
        [
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "grid_id",
            "latitude",
            "longitude",
        ],
    ].copy()
    frame["hub_u117"] = u50 * ratio
    frame["hub_v117"] = v50 * ratio

    grid_table = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .sort_values("grid_id")
    )
    if len(grid_table) != 16:
        raise ValueError(f"expected 16 LDAPS grids, found {len(grid_table)}")
    farm = np.asarray(TURBINES_BY_GROUP[TARGET], dtype=float)
    farm_lat = float(farm[:, 0].mean())
    farm_lon = float(farm[:, 1].mean())
    lat = grid_table["latitude"].to_numpy(dtype=float)
    lon = grid_table["longitude"].to_numpy(dtype=float)
    mean_lat = np.deg2rad((lat + farm_lat) / 2.0)
    grid_y = (lat - farm_lat) * 111.32
    grid_x = (lon - farm_lon) * 111.32 * np.cos(mean_lat)

    u_frame = frame.pivot(
        index="forecast_kst_dtm",
        columns="grid_id",
        values="hub_u117",
    ).sort_index()
    v_frame = frame.pivot(
        index="forecast_kst_dtm",
        columns="grid_id",
        values="hub_v117",
    ).sort_index()
    grid_ids = grid_table["grid_id"].to_list()
    u_frame = u_frame.reindex(columns=grid_ids)
    v_frame = v_frame.reindex(columns=grid_ids)
    issues = (
        frame.drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")["data_available_kst_dtm"]
        .sort_index()
    )
    if requested_index is not None:
        u_frame = u_frame.reindex(requested_index)
        v_frame = v_frame.reindex(requested_index)
        issues = issues.reindex(requested_index)
    u = u_frame.to_numpy(dtype=float)
    v = v_frame.to_numpy(dtype=float)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("LDAPS hub-vector coverage is incomplete")
    if issues.isna().any():
        raise ValueError("LDAPS issue-time coverage is incomplete")
    return u, v, grid_x, grid_y, issues


def upwind_kernel_speed(
    hub_u: np.ndarray,
    hub_v: np.ndarray,
    grid_x_km: np.ndarray,
    grid_y_km: np.ndarray,
    upwind_distance_km: float,
    *,
    sigma_km: float = KERNEL_SIGMA_KM,
) -> np.ndarray:
    hub_u = np.asarray(hub_u, dtype=float)
    hub_v = np.asarray(hub_v, dtype=float)
    grid_x_km = np.asarray(grid_x_km, dtype=float)
    grid_y_km = np.asarray(grid_y_km, dtype=float)
    if hub_u.shape != hub_v.shape or hub_u.ndim != 2:
        raise ValueError("hub-vector arrays must be aligned and 2-D")
    if hub_u.shape[1] != len(grid_x_km) or len(grid_x_km) != len(
        grid_y_km
    ):
        raise ValueError("grid coordinates are misaligned")
    if upwind_distance_km < 0.0 or sigma_km <= 0.0:
        raise ValueError("kernel distance and width must be valid")

    center_weight = np.exp(
        -(grid_x_km**2 + grid_y_km**2) / (2.0 * sigma_km**2)
    )
    center_weight /= center_weight.sum()
    direction_u = hub_u @ center_weight
    direction_v = hub_v @ center_weight
    magnitude = np.clip(np.hypot(direction_u, direction_v), 0.05, None)
    target_x = -upwind_distance_km * direction_u / magnitude
    target_y = -upwind_distance_km * direction_v / magnitude
    distance_sq = (
        (grid_x_km[None, :] - target_x[:, None]) ** 2
        + (grid_y_km[None, :] - target_y[:, None]) ** 2
    )
    weight = np.exp(-distance_sq / (2.0 * sigma_km**2))
    weight /= weight.sum(axis=1, keepdims=True)
    selected_u = np.sum(weight * hub_u, axis=1)
    selected_v = np.sum(weight * hub_v, axis=1)
    speed = np.hypot(selected_u, selected_v)
    if not np.isfinite(speed).all():
        raise ValueError("upwind kernel produced non-finite speed")
    return speed


def fit_power_curve(
    train_speed: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    predict_speed: np.ndarray,
) -> np.ndarray:
    train_speed = np.asarray(train_speed, dtype=float)
    truth = np.asarray(truth, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    if int(train_mask.sum()) < 1_000:
        raise ValueError("power curve needs at least 1,000 timestamps")
    model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    model.fit(train_speed[train_mask], truth[train_mask])
    prediction = np.asarray(model.predict(predict_speed), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("power curve produced non-finite predictions")
    return prediction


def advection_gate(
    member: np.ndarray,
    incumbent: np.ndarray,
    available: np.ndarray,
    policy: AdvectionPolicy,
) -> np.ndarray:
    difference = member - incumbent
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
        & (np.abs(difference) >= policy.minimum_disagreement_kwh)
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )


def apply_advection_policy(
    member: np.ndarray,
    incumbent: np.ndarray,
    available: np.ndarray,
    policy: AdvectionPolicy,
    *,
    maximum_incremental_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < maximum_incremental_movement_ratio <= 0.01:
        raise ValueError("incremental movement ratio must be in (0, 0.01]")
    gate = advection_gate(member, incumbent, available, policy)
    bound = maximum_incremental_movement_ratio * CAPACITY
    movement = np.clip(
        policy.alpha * (member - incumbent), -bound, bound
    )
    candidate = incumbent.copy()
    candidate[gate] = incumbent[gate] + movement[gate]
    return np.clip(candidate, 0.0, CAPACITY), gate


def select_advection_policy(
    truth: np.ndarray,
    incumbent: np.ndarray,
    members: dict[float, np.ndarray],
    available: np.ndarray,
    selection: np.ndarray,
    *,
    minimum_changed_rows: int,
    maximum_changed_ratio: float,
    maximum_incremental_movement_ratio: float,
) -> dict[str, object]:
    screened: list[dict[str, object]] = []
    positive: list[dict[str, object]] = []
    for upwind_distance, member in members.items():
        difference = member - incumbent
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
                    for alpha in (0.05, 0.10, 0.20, 0.35):
                        policy = AdvectionPolicy(
                            upwind_distance_km=upwind_distance,
                            direction=direction_name,
                            coverage=coverage,
                            minimum_base_ratio=minimum_ratio,
                            maximum_base_ratio=maximum_ratio,
                            minimum_disagreement_kwh=threshold,
                            alpha=alpha,
                        )
                        candidate, gate = apply_advection_policy(
                            member,
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
                        screened.append(record)
                        if (
                            min(
                                comparison["delta"][component]
                                for component in COMPONENTS
                            )
                            > 0.0
                        ):
                            positive.append(record)

    def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
        delta = row["comparison"]["delta"]
        return (
            float(delta["score"]),
            float(min(delta["one_minus_nmae"], delta["ficr"])),
            -float(row["mean_absolute_increment_kwh"]),
        )

    screened.sort(key=sort_key, reverse=True)
    positive.sort(key=sort_key, reverse=True)

    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {**row, "policy": row["policy"].to_dict()}

    return {
        "selected": None if not positive else positive[0],
        "top_positive": [serialize(row) for row in positive[:10]],
        "top_screened": [serialize(row) for row in screened[:10]],
        "screened_policies": len(screened),
        "positive_policies": len(positive),
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
    train_speed: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    policy: AdvectionPolicy,
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
    test_u, test_v, grid_x, grid_y, _ = load_ldaps_hub_vectors(
        test_ldaps, index
    )
    test_speed = upwind_kernel_speed(
        test_u,
        test_v,
        grid_x,
        grid_y,
        policy.upwind_distance_km,
    )
    member = fit_power_curve(
        train_speed, truth, train_mask, test_speed
    )
    incumbent = incumbent_frame[TARGET].to_numpy(dtype=float)
    candidate, gate = apply_advection_policy(
        member,
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
    issues = pd.to_datetime(cache["issue_ns"]).to_numpy()
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
    issue_error_hours = float(
        np.max(
            np.abs(
                (
                    pd.to_datetime(ldaps_issues.to_numpy())
                    - pd.to_datetime(issues)
                )
                / np.timedelta64(1, "h")
            )
        )
    )
    if issue_error_hours > 0.0:
        raise ValueError("LDAPS issue times do not match the OOF cache")
    speeds = {
        distance: upwind_kernel_speed(
            hub_u, hub_v, grid_x, grid_y, distance
        )
        for distance in UPWIND_DISTANCES_KM
    }
    q1_members = {
        distance: fit_power_curve(speed, truth, q1, speed)
        for distance, speed in speeds.items()
    }
    selection = select_advection_policy(
        truth,
        static_incumbent,
        q1_members,
        available,
        q2,
        minimum_changed_rows=args.minimum_selection_rows,
        maximum_changed_ratio=args.maximum_changed_ratio,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    report: dict[str, Any] = {
        "family": "ldaps_upwind_advection_power_curve",
        "method": {
            "wind_field": "117 m power-law-extrapolated LDAPS hub vectors",
            "kernel": (
                "Gaussian grid weights centered along the instantaneous "
                "upwind direction from the group-3 turbine centroid"
            ),
            "kernel_sigma_km": KERNEL_SIGMA_KM,
            "predeclared_upwind_distances_km": list(
                UPWIND_DISTANCES_KM
            ),
            "power_mapping": "monotone isotonic wind-to-power curve",
        },
        "contract": {
            "q1_model_train": "2024-01-01 through 2024-03-31",
            "q2_family_and_policy_selection": (
                "2024-04-01 through 2024-07-01 00:00"
            ),
            "locked_h2": (
                "selected distance with H1 refit; 2024-07-01 01:00 "
                "through 2024-12-31 23:00"
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
            "grids": hub_u.shape[1],
            "q1_rows": int(q1.sum()),
            "q2_rows": int(q2.sum()),
            "h1_rows": int(h1.sum()),
            "h2_rows": int(h2.sum()),
            "issue_time_max_error_hours": issue_error_hours,
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
                "no upwind-kernel policy improved every component on Q2"
            ),
        }
        report["submission"] = None
        _write_json(Path(args.output), report)
        return report

    policy = selection["selected"]["policy"]
    selected_speed = speeds[policy.upwind_distance_km]
    static_member = q1_members[policy.upwind_distance_km]
    static_candidate, static_gate = apply_advection_policy(
        static_member,
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
    h1_member = fit_power_curve(
        selected_speed, truth, h1, selected_speed
    )
    locked_candidate, locked_gate = apply_advection_policy(
        h1_member,
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
            train_speed=selected_speed,
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
            "upwind-advection candidate passed every locked gate"
            if qualified
            else "upwind-advection candidate failed at least one locked gate"
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
            "ldaps_upwind_advection_power_curve_20260726.json"
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
            "ldaps_upwind_advection_power_curve_20260726.csv"
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
