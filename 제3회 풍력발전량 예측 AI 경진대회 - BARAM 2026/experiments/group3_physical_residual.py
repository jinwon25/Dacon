from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.exact_oof_meta_gate import actionable_mask, apply_meta_gate
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from experiments.group3_physical_catboost import _bootstrap_days, _compare
from src.metrics import CAPACITY_KWH, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
FINE_THRESHOLD = 0.545
FINE_ALPHA = 0.50


@dataclass(frozen=True)
class Policy:
    family: str
    alpha: float
    min_disagreement_ratio: float
    max_disagreement_ratio: float
    max_seed_std_ratio: float


def _column(frame: pd.DataFrame, source: str, variable: str) -> pd.Series:
    name = f"{source}__{TARGET}__{variable}__idw"
    if name not in frame:
        raise KeyError(f"Required physical feature is missing: {name}")
    return frame[name].astype(float)


def _wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.sqrt(u * u + v * v).clip(lower=0.05)


def _shear(lower: pd.Series, upper: pd.Series, z0: float, z1: float) -> pd.Series:
    value = np.log(upper.clip(lower=0.05) / lower.clip(lower=0.05)) / np.log(z1 / z0)
    return value.replace([np.inf, -np.inf], np.nan).fillna(0.14).clip(-0.30, 0.60)


def _veer(
    lower_u: pd.Series,
    lower_v: pd.Series,
    upper_u: pd.Series,
    upper_v: pd.Series,
) -> pd.Series:
    cross = lower_u * upper_v - lower_v * upper_u
    dot = lower_u * upper_u + lower_v * upper_v
    return pd.Series(np.arctan2(cross, dot), index=lower_u.index).clip(-np.pi, np.pi)


def _air_density(pressure_pa: pd.Series, temperature_k: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Moist-air density from the ideal-gas mixture, bounded to plausible surface values."""
    temperature_c = temperature_k - 273.15
    saturation_vapor_pressure = 611.2 * np.exp(
        17.67 * temperature_c / (temperature_c + 243.5)
    )
    vapor_pressure = (rh_pct.clip(0.0, 100.0) / 100.0) * saturation_vapor_pressure
    dry_pressure = (pressure_pa - vapor_pressure).clip(lower=1.0)
    density = dry_pressure / (287.05 * temperature_k.clip(lower=200.0))
    density += vapor_pressure / (461.495 * temperature_k.clip(lower=200.0))
    return density.clip(0.80, 1.40)


def build_physical_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build compact group-3 physics without using labels or neighboring targets."""
    ldaps_u10 = _column(frame, "ldaps", "heightAboveGround_10_10u")
    ldaps_v10 = _column(frame, "ldaps", "heightAboveGround_10_10v")
    # LDAPS does not retain turbine-IDW 50 m vector components in the compact
    # cache. Its cached hub vector is extrapolated from their mean, so it is the
    # leakage-free direction proxy paired with the independently retained 50 m
    # speed extrema below.
    ldaps_hub_u = _column(frame, "ldaps", "hub_u117")
    ldaps_hub_v = _column(frame, "ldaps", "hub_v117")
    gfs_u80 = _column(frame, "gfs", "heightAboveGround_80_u")
    gfs_v80 = _column(frame, "gfs", "heightAboveGround_80_v")
    gfs_u100 = _column(frame, "gfs", "heightAboveGround_100_100u")
    gfs_v100 = _column(frame, "gfs", "heightAboveGround_100_100v")

    ldaps_ws10 = _wind_speed(ldaps_u10, ldaps_v10)
    ldaps_ws50 = (
        _column(frame, "ldaps", "ws50_max")
        + _column(frame, "ldaps", "ws50_min")
    ) / 2.0
    gfs_ws80 = _wind_speed(gfs_u80, gfs_v80)
    gfs_ws100 = _wind_speed(gfs_u100, gfs_v100)
    ldaps_hub = _column(frame, "ldaps", "hub_ws117")
    gfs_hub = _column(frame, "gfs", "hub_ws117")

    ldaps_density = _air_density(
        _column(frame, "ldaps", "surface_0_sp"),
        _column(frame, "ldaps", "heightAboveGround_2_t"),
        _column(frame, "ldaps", "heightAboveGround_2_r"),
    )
    gfs_density = _air_density(
        _column(frame, "gfs", "surface_0_sp"),
        _column(frame, "gfs", "heightAboveGround_2_2t"),
        _column(frame, "gfs", "heightAboveGround_2_2r"),
    )
    ldaps_density_ws = ldaps_hub * (ldaps_density / 1.225) ** (1.0 / 3.0)
    gfs_density_ws = gfs_hub * (gfs_density / 1.225) ** (1.0 / 3.0)
    gfs_gust = _column(frame, "gfs", "surface_0_gust")

    out = pd.DataFrame(index=frame.index)
    out["ldaps_hub_ws"] = ldaps_hub
    out["gfs_hub_ws"] = gfs_hub
    out["hub_ws_mean"] = (ldaps_hub + gfs_hub) / 2.0
    out["hub_ws_abs_delta"] = np.abs(ldaps_hub - gfs_hub)
    out["hub_ws_rel_delta"] = np.abs(ldaps_hub - gfs_hub) / (
        (ldaps_hub + gfs_hub) / 2.0
    ).clip(lower=0.5)
    out["ldaps_density_ws"] = ldaps_density_ws
    out["gfs_density_ws"] = gfs_density_ws
    out["density_ws_mean"] = (ldaps_density_ws + gfs_density_ws) / 2.0
    out["ldaps_power_density"] = 0.5 * ldaps_density * ldaps_hub.pow(3)
    out["gfs_power_density"] = 0.5 * gfs_density * gfs_hub.pow(3)

    out["ldaps_shear"] = _shear(ldaps_ws10, ldaps_ws50, 10.0, 50.0)
    out["gfs_shear"] = _shear(gfs_ws80, gfs_ws100, 80.0, 100.0)
    out["ldaps_veer"] = _veer(
        ldaps_u10, ldaps_v10, ldaps_hub_u, ldaps_hub_v
    )
    out["gfs_veer"] = _veer(gfs_u80, gfs_v80, gfs_u100, gfs_v100)
    out["ldaps_veer_abs"] = np.abs(out["ldaps_veer"])
    out["gfs_veer_abs"] = np.abs(out["gfs_veer"])

    out["ldaps_density"] = ldaps_density
    out["gfs_density"] = gfs_density
    out["density_delta"] = ldaps_density - gfs_density
    out["gfs_gust_factor"] = gfs_gust / gfs_hub.clip(lower=0.5)
    out["gfs_gust_excess"] = gfs_gust - gfs_hub
    out["lead_hour"] = frame["lead_hour"].astype(float)
    out["hour_sin"] = frame["hour_sin"].astype(float)
    out["hour_cos"] = frame["hour_cos"].astype(float)
    out["doy_sin"] = frame["doy_sin"].astype(float)
    out["doy_cos"] = frame["doy_cos"].astype(float)

    hub = [
        "current_ratio",
        "ldaps_hub_ws",
        "gfs_hub_ws",
        "hub_ws_mean",
        "hub_ws_abs_delta",
        "hub_ws_rel_delta",
        "ldaps_density_ws",
        "gfs_density_ws",
        "density_ws_mean",
        "ldaps_power_density",
        "gfs_power_density",
    ]
    shear = [*hub, "ldaps_shear", "gfs_shear", "ldaps_veer", "gfs_veer"]
    full = [
        *shear,
        "ldaps_veer_abs",
        "gfs_veer_abs",
        "ldaps_density",
        "gfs_density",
        "density_delta",
        "gfs_gust_factor",
        "gfs_gust_excess",
        "lead_hour",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    return out.astype("float32"), {"hub_density": hub, "shear_veer": shear, "full_regime": full}


def _model(seed: int, iterations: int = 650) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=iterations,
        learning_rate=0.025,
        depth=5,
        l2_leaf_reg=14.0,
        random_seed=seed,
        random_strength=0.35,
        bootstrap_type="Bayesian",
        bagging_temperature=0.35,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def fit_residual_members(
    physical: pd.DataFrame,
    columns: list[str],
    truth: np.ndarray,
    current: np.ndarray,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    seeds: tuple[int, ...],
) -> np.ndarray:
    design = physical.copy()
    design["current_ratio"] = np.asarray(current, dtype=float) / CAPACITY
    eligible_train = np.asarray(train_mask, dtype=bool) & (np.asarray(truth) >= 0.10 * CAPACITY)
    residual = (np.asarray(truth, dtype=float) - np.asarray(current, dtype=float)) / CAPACITY
    predictions = []
    for seed in seeds:
        model = _model(seed)
        model.fit(design.loc[eligible_train, columns], residual[eligible_train])
        correction = np.clip(model.predict(design.loc[predict_mask, columns]), -0.08, 0.08)
        member = np.clip(current[predict_mask] + correction * CAPACITY, 0.0, CAPACITY)
        predictions.append(member)
    return np.column_stack(predictions)


def apply_policy(
    current: np.ndarray,
    seed_predictions: np.ndarray,
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    seed_predictions = np.asarray(seed_predictions, dtype=float)
    mean_member = seed_predictions.mean(axis=1)
    seed_delta = seed_predictions - current[:, None]
    unanimous = np.all(seed_delta >= 0.0, axis=1) | np.all(seed_delta <= 0.0, axis=1)
    disagreement = np.abs(mean_member - current) / CAPACITY
    uncertainty = seed_predictions.std(axis=1) / CAPACITY
    gate = (
        (current >= 0.10 * CAPACITY)
        & unanimous
        & (disagreement >= policy.min_disagreement_ratio)
        & (disagreement <= policy.max_disagreement_ratio)
        & (uncertainty <= policy.max_seed_std_ratio)
    )
    candidate = current.copy()
    candidate[gate] = np.clip(
        current[gate] + policy.alpha * (mean_member[gate] - current[gate]),
        0.0,
        CAPACITY,
    )
    return candidate, gate


def _period_delta(
    truth: np.ndarray, current: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    return _compare(truth[mask], current[mask], candidate[mask])["delta"]


def select_policy(
    truth: np.ndarray,
    current: np.ndarray,
    members_by_family: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    selection_mask: np.ndarray,
) -> tuple[Policy, list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for family, seed_predictions in members_by_family.items():
        for alpha in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
            for minimum in (0.0, 0.0025, 0.005):
                for maximum in (0.02, 0.04, 0.06):
                    if minimum >= maximum:
                        continue
                    for max_std in (0.0025, 0.005, 0.01):
                        policy = Policy(family, alpha, minimum, maximum, max_std)
                        candidate, gate = apply_policy(current, seed_predictions, policy)
                        coverage = float(gate[selection_mask].mean())
                        if coverage < 0.01 or coverage > 0.35:
                            continue
                        delta = _period_delta(truth, current, candidate, selection_mask)
                        monthly = {}
                        for month in (4, 5, 6):
                            month_mask = selection_mask & (index.month == month)
                            monthly[str(month)] = _period_delta(
                                truth, current, candidate, month_mask
                            )["score"]
                        seed_deltas = []
                        for column in range(seed_predictions.shape[1]):
                            one_seed = np.repeat(
                                seed_predictions[:, column : column + 1],
                                seed_predictions.shape[1],
                                axis=1,
                            )
                            seed_candidate, _ = apply_policy(current, one_seed, policy)
                            seed_deltas.append(
                                _period_delta(
                                    truth, current, seed_candidate, selection_mask
                                )["score"]
                            )
                        rows.append(
                            {
                                "policy": asdict(policy),
                                "coverage": coverage,
                                "changed_rows": int((gate & selection_mask).sum()),
                                "delta": delta,
                                "monthly_score_deltas": monthly,
                                "months_improved": int(sum(value > 0 for value in monthly.values())),
                                "seed_score_deltas": seed_deltas,
                                "min_seed_score_delta": float(min(seed_deltas)),
                            }
                        )
    eligible = [
        row
        for row in rows
        if row["delta"]["one_minus_nmae"] > 0.0
        and row["delta"]["ficr"] > 0.0
        and row["months_improved"] >= 2
        and row["min_seed_score_delta"] > 0.0
    ]
    pool = eligible if eligible else rows
    if not pool:
        raise RuntimeError("No bounded physical residual policy had usable Q2 coverage")
    ranked = sorted(
        pool,
        key=lambda row: (
            row["delta"]["one_minus_nmae"] > 0.0
            and row["delta"]["ficr"] > 0.0,
            row["months_improved"],
            row["min_seed_score_delta"],
            row["delta"]["score"],
            -row["coverage"],
        ),
        reverse=True,
    )
    return Policy(**ranked[0]["policy"]), ranked[:30], bool(eligible)


def _fine_validation_baseline(
    labels_path: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    (
        _labels,
        index,
        truth,
        group_1,
        group_2,
        base,
        member,
        current,
        _features,
        _action,
    ) = _prepare_validation(labels_path, driver_cache_path)
    meta = np.load(meta_cache_path, allow_pickle=True)
    meta_index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    if not index.equals(meta_index):
        raise ValueError("Meta and exact-driver validation indexes differ")
    action = actionable_mask(group_1, group_2, base, member) & (truth >= 0.10 * CAPACITY)
    fine, _ = apply_meta_gate(
        current,
        member,
        action,
        meta["valid_probability"].astype(float),
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    return index, truth, fine


def _write_submission(
    output_path: Path,
    base_submission_path: Path,
    test_features_path: Path,
    train_physical: pd.DataFrame,
    family_columns: list[str],
    validation_truth: np.ndarray,
    validation_current: np.ndarray,
    policy: Policy,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    base = pd.read_csv(base_submission_path, encoding="utf-8-sig")
    base_index = pd.DatetimeIndex(pd.to_datetime(base["forecast_kst_dtm"]))
    test_frame = pd.read_pickle(test_features_path)
    if not base_index.equals(test_frame.index):
        raise ValueError("Test feature cache does not match the base submission")
    test_physical, _ = build_physical_features(test_frame)
    combined_physical = pd.concat([train_physical, test_physical], axis=0)
    combined_truth = np.concatenate(
        [validation_truth, np.full(len(test_physical), np.nan, dtype=float)]
    )
    test_current = base[TARGET].to_numpy(dtype=float)
    combined_current = np.concatenate([validation_current, test_current])
    train_mask = np.arange(len(combined_physical)) < len(train_physical)
    predict_mask = ~train_mask
    test_members = fit_residual_members(
        combined_physical,
        family_columns,
        combined_truth,
        combined_current,
        train_mask,
        predict_mask,
        seeds,
    )
    candidate, gate = apply_policy(test_current, test_members, policy)
    output = base.copy()
    output[TARGET] = candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    payload = output_path.read_bytes()
    movement = candidate - test_current
    return {
        "path": str(output_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "changed_rows": int(gate.sum()),
        "changed_ratio": float(gate.mean()),
        "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
        "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
        "max_absolute_movement_kwh": float(np.abs(movement).max()),
        "groups_1_2_unchanged": bool(
            np.array_equal(output["kpx_group_1"], base["kpx_group_1"])
            and np.array_equal(output["kpx_group_2"], base["kpx_group_2"])
        ),
    }


def run_experiment(
    feature_cache_path: Path,
    test_feature_cache_path: Path,
    labels_path: Path,
    driver_cache_path: Path,
    meta_cache_path: Path,
    base_submission_path: Path,
    output_path: Path,
    artifact_dir: Path,
    evaluation_output: Path,
    seeds: tuple[int, ...],
    n_bootstrap: int,
) -> dict[str, object]:
    started = time.perf_counter()
    cached = pd.read_pickle(feature_cache_path)
    index, truth, current = _fine_validation_baseline(
        labels_path, driver_cache_path, meta_cache_path
    )
    aligned = cached.reindex(index)
    if aligned.isna().all(axis=1).any():
        raise ValueError("Feature cache does not cover the exact validation index")
    physical, feature_families = build_physical_features(aligned)

    q1 = (index >= pd.Timestamp("2024-01-01")) & (index < Q2_START)
    q2 = (index >= Q2_START) & (index < H2_START)
    h1 = index < H2_START
    h2 = index >= H2_START

    q2_members = {
        family: fit_residual_members(
            physical, columns, truth, current, q1, q2, seeds
        )
        for family, columns in feature_families.items()
    }
    q2_current = current[q2]
    q2_truth = truth[q2]
    q2_index = index[q2]
    selected_policy, development_top, development_qualified = select_policy(
        q2_truth,
        q2_current,
        q2_members,
        q2_index,
        np.ones(len(q2_index), dtype=bool),
    )

    selected_columns = feature_families[selected_policy.family]
    h2_members = fit_residual_members(
        physical, selected_columns, truth, current, h1, h2, seeds
    )
    h2_current = current[h2]
    h2_truth = truth[h2]
    h2_index = index[h2]
    h2_candidate, h2_gate = apply_policy(h2_current, h2_members, selected_policy)
    locked_comparison = _compare(h2_truth, h2_current, h2_candidate)

    monthly = {}
    for month in range(7, 13):
        mask = h2_index.month == month
        monthly[str(month)] = _period_delta(
            h2_truth, h2_current, h2_candidate, mask
        )
    seed_deltas = []
    for column, seed in enumerate(seeds):
        one_seed = np.repeat(
            h2_members[:, column : column + 1], h2_members.shape[1], axis=1
        )
        seed_candidate, _ = apply_policy(h2_current, one_seed, selected_policy)
        seed_deltas.append(
            {
                "seed": seed,
                "delta": _compare(h2_truth, h2_current, seed_candidate)["delta"],
            }
        )
    bootstrap = _bootstrap_days(
        h2_truth, h2_current, h2_candidate, h2_index, n_bootstrap
    )
    delta = locked_comparison["delta"]
    positive_months = int(sum(row["score"] > 0.0 for row in monthly.values()))
    strict_gates = {
        "development_contract_passed": development_qualified,
        "locked_score_delta_at_least_0_0005": delta["score"] >= 0.0005,
        "locked_one_minus_nmae_positive": delta["one_minus_nmae"] > 0.0,
        "locked_ficr_positive": delta["ficr"] > 0.0,
        "at_least_four_positive_months": positive_months >= 4,
        "all_seed_directions_positive": all(
            row["delta"]["score"] > 0.0 for row in seed_deltas
        ),
        "bootstrap_positive_fraction_at_least_0_80": bootstrap["positive_fraction"] >= 0.80,
        "bootstrap_q05_above_minus_0_0005": bootstrap["q05"] >= -0.0005,
    }
    qualified = bool(all(strict_gates.values()))
    submission = None
    if qualified:
        submission = _write_submission(
            output_path,
            base_submission_path,
            test_feature_cache_path,
            physical,
            selected_columns,
            truth,
            current,
            selected_policy,
            seeds,
        )

    movement = h2_candidate - h2_current
    report: dict[str, object] = {
        "method": "bounded group-3 physical residual correction",
        "hypothesis": (
            "Hub-height interpolation, vertical shear/veer, moist-air density and "
            "power-density regimes explain residual error left by the public-best ensemble."
        ),
        "selection_contract": {
            "development": "2024-Q1 train -> 2024-Q2 select",
            "locked": "2024-H1 train -> 2024-H2 evaluate once",
            "public_best_baseline": "fine meta-gate threshold=0.545 alpha=0.50",
            "families": {name: columns for name, columns in feature_families.items()},
            "seeds": list(seeds),
        },
        "selected_policy": asdict(selected_policy),
        "development_qualified": development_qualified,
        "development_top": development_top,
        "locked_h2": {
            "comparison": locked_comparison,
            "monthly_deltas": monthly,
            "positive_months": positive_months,
            "seed_deltas": seed_deltas,
            "day_bootstrap": bootstrap,
        },
        "movement": {
            "changed_rows": int(h2_gate.sum()),
            "changed_ratio": float(h2_gate.mean()),
            "mean_absolute_kwh": float(np.abs(movement).mean()),
            "p95_absolute_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "max_absolute_kwh": float(np.abs(movement).max()),
        },
        "strict_promotion_gates": strict_gates,
        "qualified": qualified,
        "submission": submission,
        "runtime_seconds": time.perf_counter() - started,
        "decision": (
            "Promote the strictly qualified physical-residual candidate."
            if qualified
            else "Reject without creating a submission CSV; retain only compact diagnostics."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "physical_residual_cache.npz",
        h2_index_ns=h2_index.astype("int64").to_numpy(),
        h2_truth=h2_truth.astype("float32"),
        h2_current=h2_current.astype("float32"),
        h2_candidate=h2_candidate.astype("float32"),
        h2_gate=h2_gate,
    )
    (artifact_dir / "physical_residual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation = {
        "family": "group3_physical_residual",
        "locked_score_delta": delta["score"],
        "locked_one_minus_nmae_delta": delta["one_minus_nmae"],
        "locked_ficr_delta": delta["ficr"],
        "expected_macro_score_delta": delta["score"] / 3.0,
        "positive_months": positive_months,
        "total_months": 6,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "bootstrap_q05": bootstrap["q05"],
        "changed_ratio": float(h2_gate.mean()),
        "p95_movement_ratio": float(np.quantile(np.abs(movement), 0.95) / CAPACITY),
        "fold_scores": [row["delta"]["score"] for row in seed_deltas],
        "cv_mean": float(np.mean([row["delta"]["score"] for row in seed_deltas])),
        "cv_std": float(np.std([row["delta"]["score"] for row in seed_deltas])),
        "oof_path": str(artifact_dir / "physical_residual_cache.npz"),
        "runtime_seconds": report["runtime_seconds"],
        "leakage_risk": "low",
        "rule_violation": "none",
        "selection_metric": delta["score"] / 3.0,
        "selection_direction": "maximize",
        "qualified": qualified,
        "submission_path": None if submission is None else submission["path"],
    }
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache", default="artifacts_final/feature_cache/features_train.pkl"
    )
    parser.add_argument(
        "--test-feature-cache", default="artifacts_final/feature_cache/features_test.pkl"
    )
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--base-submission", default="submissions/blend_best_crossg3_traj_meta_finesweep.csv"
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_g3_physical_residual.csv"
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts_final/physical_residual"
    )
    parser.add_argument(
        "--agent-evaluation-output",
        default="artifacts_final/physical_residual/agent_evaluation.json",
    )
    parser.add_argument("--seeds", default="63001,63002,63003")
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = run_experiment(
        Path(args.feature_cache),
        Path(args.test_feature_cache),
        Path(args.labels),
        Path(args.driver_cache),
        Path(args.meta_cache),
        Path(args.base_submission),
        Path(args.output),
        Path(args.artifact_dir),
        Path(args.agent_evaluation_output),
        seeds,
        args.n_bootstrap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
