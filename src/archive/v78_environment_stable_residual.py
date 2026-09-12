"""Preregistered environment-stable trajectory residual above strong OOF parents.

Feature selection is repeated inside each source origin.  A feature survives
only when its standardized residual coefficient keeps the same direction in
at least the frozen fraction of target-free domain x month-band environments.
Environment target means are removed before fitting, so the model cannot act
as a carried-forward domain/month base-rate lookup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes
from src.archive.v58_eta15_rebase_audit import CURRENT_ETA, eta_parent
from src.archive.v61_final_gate_oof import _source_global_rate, apply_final_gate


PROTOCOL = "V78_ENVIRONMENT_STABLE_TRAJECTORY_RESIDUAL_V1"
TARGET = "control_success"


@dataclass(frozen=True)
class StableSpec:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    risk: str


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return np.where(np.isfinite(values), values, float(default))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen ID-free row-local feature family."""

    p_n = np.maximum(_numeric(frame, "asof_pitcher_n"), 0.0)
    b_n = np.maximum(_numeric(frame, "asof_batter_n"), 0.0)
    m_n = np.maximum(_numeric(frame, "asof_pitcher_pitchmix_n"), 0.0)
    career = _numeric(frame, "asof_pitcher_success_rate", 0.5)
    prev1 = _numeric(frame, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(frame, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(frame, "asof_pitcher_prev5_game_success_rate", 0.5)
    middle1 = _numeric(frame, "asof_pitcher_prev1_game_middle_rate", 0.0)
    middle3 = _numeric(frame, "asof_pitcher_prev3_game_middle_rate", 0.0)
    middle5 = _numeric(frame, "asof_pitcher_prev5_game_middle_rate", 0.0)
    strike = _numeric(frame, "asof_pitcher_strike_rate", 0.0)
    ball = _numeric(frame, "asof_pitcher_ball_rate", 0.0)
    middle = _numeric(frame, "asof_pitcher_middle_rate", 0.0)
    reverse = _numeric(frame, "asof_pitcher_reverse_rate", 0.0)
    balls = _numeric(frame, "balls_before", 0.0)
    strikes = _numeric(frame, "strikes_before", 0.0)
    pressure = ((balls == 3.0) | (strikes == 2.0)).astype(np.float64)
    pitcher_hand = frame["pitcher_hand"].astype("string").fillna("__MISSING__")
    batter_hand = frame["batter_hand"].astype("string").fillna("__MISSING__")

    mix = np.column_stack(
        [
            _numeric(frame, "asof_pitcher_fastball_rate", 1.0 / 3.0),
            _numeric(frame, "asof_pitcher_breaking_rate", 1.0 / 3.0),
            _numeric(frame, "asof_pitcher_offspeed_rate", 1.0 / 3.0),
        ]
    )
    mix = np.where(np.isfinite(mix), mix, 1.0 / 3.0)
    mix = np.clip(mix, 1e-6, None)
    mix /= mix.sum(axis=1, keepdims=True)
    recent_delta = 0.50 * prev1 + 0.30 * prev3 + 0.20 * prev5 - career

    output = pd.DataFrame(
        {
            "pitcher_prev1_minus_career": prev1 - career,
            "pitcher_prev3_minus_career": prev3 - career,
            "pitcher_prev5_minus_career": prev5 - career,
            "pitcher_prev1_minus_prev3": prev1 - prev3,
            "pitcher_prev3_minus_prev5": prev3 - prev5,
            "pitcher_success_curvature": prev1 - 2.0 * prev3 + prev5,
            "middle_prev1_minus_prev3": middle1 - middle3,
            "middle_prev3_minus_prev5": middle3 - middle5,
            "middle_curvature": middle1 - 2.0 * middle3 + middle5,
            "pitcher_log_n": np.log1p(p_n),
            "pitcher_reliability": p_n / (p_n + 160.0),
            "batter_log_n": np.log1p(b_n),
            "batter_reliability": b_n / (b_n + 160.0),
            "pitchmix_reliability": m_n / (m_n + 160.0),
            "strike_ball_margin": strike - ball,
            "middle_reverse_margin": middle - reverse,
            "pitchmix_entropy": -np.sum(mix * np.log(mix), axis=1),
            "balls_scaled": balls / 3.0,
            "strikes_scaled": strikes / 2.0,
            "outs_scaled": _numeric(frame, "outs_before") / 2.0,
            "runners_scaled": _numeric(frame, "num_runners_on") / 3.0,
            "inning_scaled": np.clip(_numeric(frame, "inning"), 1.0, 12.0) / 9.0,
            "abs_score_diff_scaled": np.minimum(
                np.abs(_numeric(frame, "score_diff_pitcher_team")), 10.0
            )
            / 10.0,
            "log_li": np.log1p(np.maximum(_numeric(frame, "li"), 0.0)),
            "win_expectancy_gap": _numeric(frame, "home_win_expectancy", 0.5)
            - _numeric(frame, "away_win_expectancy", 0.5),
            "pressure": pressure,
            "same_hand": (pitcher_hand.to_numpy() == batter_hand.to_numpy()).astype(float),
            "pressure_x_recent_delta": pressure * recent_delta,
        }
    )
    if not np.isfinite(output.to_numpy(np.float64)).all():
        raise ValueError("non-finite stable trajectory feature")
    return output


def environment_labels(frame: pd.DataFrame) -> np.ndarray:
    """Create target-free domain x coarse calendar environments."""

    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int16)
    band = np.where(month <= 5, "early", np.where(month <= 7, "mid", "late"))
    return (
        frame["domain3"].astype("string").fillna("__MISSING__").astype(str).to_numpy()
        + "|"
        + band
    )


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return (x - means) / scales, means, scales


def _ridge(
    x: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("invalid ridge arrays")
    if x.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    if sample_weight is None:
        weight = np.ones(len(x), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
    if weight.shape != (len(x),) or np.any(weight <= 0.0):
        raise ValueError("sample weights must be positive and aligned")
    weighted_x = x * weight[:, None]
    gram = x.T @ weighted_x
    rhs = x.T @ (weight * y)
    penalty = float(alpha) * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(gram + penalty, rhs)


def _center_residual(residual: np.ndarray, environments: np.ndarray) -> np.ndarray:
    centered = np.asarray(residual, dtype=np.float64).copy()
    for environment in np.unique(environments):
        mask = environments == environment
        centered[mask] -= float(np.mean(centered[mask]))
    return centered


def select_stable_features(
    x: np.ndarray,
    residual: np.ndarray,
    environments: np.ndarray,
    feature_names: list[str],
    *,
    alpha: float,
    minimum_environment_rows: int,
    minimum_sign_consistency: float,
    minimum_median_abs_coefficient: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select coefficient directions stable across source environments only."""

    standardized, _, _ = _standardize(np.asarray(x, dtype=np.float64))
    centered = _center_residual(residual, environments)
    coefficients = []
    used = []
    for environment in sorted(np.unique(environments)):
        mask = environments == environment
        if int(mask.sum()) < int(minimum_environment_rows):
            continue
        coefficients.append(_ridge(standardized[mask], centered[mask], alpha))
        used.append(str(environment))
    if len(coefficients) < 3:
        return np.zeros(x.shape[1], dtype=bool), {
            "used_environments": used,
            "reason": "fewer_than_three_eligible_environments",
            "selected_features": [],
        }
    matrix = np.vstack(coefficients)
    positive = np.mean(matrix > 0.0, axis=0)
    negative = np.mean(matrix < 0.0, axis=0)
    consistency = np.maximum(positive, negative)
    median_abs = np.median(np.abs(matrix), axis=0)
    selected = (consistency >= float(minimum_sign_consistency)) & (
        median_abs >= float(minimum_median_abs_coefficient)
    )
    return selected, {
        "used_environments": used,
        "environment_count": int(len(used)),
        "selected_features": [
            name for name, keep in zip(feature_names, selected, strict=True) if keep
        ],
        "sign_consistency": {
            name: float(value)
            for name, value in zip(feature_names, consistency, strict=True)
        },
        "median_abs_standardized_coefficient": {
            name: float(value) for name, value in zip(feature_names, median_abs, strict=True)
        },
    }


def _risk_weights(
    risk: str, residual: np.ndarray, environments: np.ndarray
) -> np.ndarray:
    if risk == "uniform":
        return np.ones(len(residual), dtype=np.float64)
    unique, counts = np.unique(environments, return_counts=True)
    inverse = {name: 1.0 / count for name, count in zip(unique, counts, strict=True)}
    weight = np.asarray([inverse[name] for name in environments], dtype=np.float64)
    if risk == "environment_equal":
        return weight / weight.mean()
    if risk != "worst_third":
        raise ValueError(f"unknown risk: {risk}")
    losses = {
        name: float(np.mean(np.square(residual[environments == name]))) for name in unique
    }
    worst_count = max(1, int(np.ceil(len(unique) / 3.0)))
    worst = set(sorted(unique, key=lambda name: losses[name], reverse=True)[:worst_count])
    multiplier = np.asarray([3.0 if name in worst else 1.0 for name in environments])
    weight *= multiplier
    return weight / weight.mean()


def fit_stable_spec(
    features: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    environments: np.ndarray,
    risk: str,
    config: dict[str, Any],
) -> tuple[StableSpec, dict[str, Any]]:
    x = features.to_numpy(np.float64)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    selected, selection = select_stable_features(
        x,
        residual,
        environments,
        list(features.columns),
        alpha=float(config["ridge_alpha"]),
        minimum_environment_rows=int(config["minimum_environment_rows"]),
        minimum_sign_consistency=float(config["minimum_sign_consistency"]),
        minimum_median_abs_coefficient=float(
            config["minimum_median_abs_standardized_coefficient"]
        ),
    )
    standardized, means, scales = _standardize(x)
    centered = _center_residual(residual, environments)
    weight = _risk_weights(risk, residual, environments)
    coefficients = _ridge(
        standardized[:, selected],
        centered,
        float(config["ridge_alpha"]),
        weight,
    )
    spec = StableSpec(
        feature_names=tuple(features.columns[selected]),
        means=means[selected],
        scales=scales[selected],
        coefficients=coefficients,
        risk=risk,
    )
    return spec, selection


def predict_correction(
    spec: StableSpec, features: pd.DataFrame, correction_cap: float
) -> np.ndarray:
    if not spec.feature_names:
        return np.zeros(len(features), dtype=np.float64)
    x = features.loc[:, list(spec.feature_names)].to_numpy(np.float64)
    correction = ((x - spec.means) / spec.scales) @ spec.coefficients
    return np.clip(correction, -float(correction_cap), float(correction_cap))


def crossfit_source(
    frame: pd.DataFrame,
    parent: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features = build_features(frame)
    target = frame["target"].to_numpy(np.float64)
    environments = environment_labels(frame)
    corrections = {
        risk: np.zeros(len(frame), dtype=np.float64) for risk in config["risks"]
    }
    fold_audits: dict[str, Any] = {}
    eligible = [
        environment
        for environment in sorted(np.unique(environments))
        if int(np.sum(environments == environment)) >= int(config["minimum_environment_rows"])
    ]
    if len(eligible) < 4:
        return corrections, {"reason": "fewer_than_four_crossfit_environments"}
    covered = np.zeros(len(frame), dtype=bool)
    for held in eligible:
        valid = environments == held
        train = ~valid
        covered |= valid
        fold_audits[held] = {}
        for risk in config["risks"]:
            spec, selection = fit_stable_spec(
                features.loc[train].reset_index(drop=True),
                target[train],
                np.asarray(parent)[train],
                environments[train],
                str(risk),
                config,
            )
            corrections[str(risk)][valid] = predict_correction(
                spec,
                features.loc[valid].reset_index(drop=True),
                float(config["correction_cap"]),
            )
            fold_audits[held][str(risk)] = {
                "selected_feature_count": int(len(spec.feature_names)),
                "selected_features": list(spec.feature_names),
                "selection_environment_count": int(selection.get("environment_count", 0)),
            }
    if not covered.all():
        # Small source environments are never used to choose the recipe.
        for risk in corrections:
            corrections[risk][~covered] = 0.0
    return corrections, {
        "eligible_environments": eligible,
        "covered_rows": int(covered.sum()),
        "uncovered_small_environment_rows": int((~covered).sum()),
        "folds": fold_audits,
    }


def _source_recipe(
    frame: pd.DataFrame,
    parent: np.ndarray,
    corrections: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    active = np.ones(len(frame), dtype=bool)
    for risk in config["risks"]:
        for eta in config["eta_grid"]:
            candidate = np.clip(
                np.asarray(parent) + float(eta) * corrections[str(risk)], 0.001, 0.999
            )
            result = diagnostics(frame, parent, candidate, active)
            rows.append(
                {
                    "risk": str(risk),
                    "eta": float(eta),
                    "gain": float(result["gain"]),
                    "positive_month_fraction": float(result["positive_month_fraction"]),
                    "worst_month_gain": float(result["worst_month_gain"]),
                    "minimum_domain_gain": float(result["minimum_domain_gain"]),
                    "mean_abs_shift": float(result["mean_abs_shift"]),
                }
            )
    metrics = pd.DataFrame(rows)
    nonzero = metrics["eta"].gt(0.0)
    metrics["passes_source_gate"] = (
        nonzero
        & metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].ge(0.75)
        & metrics["worst_month_gain"].gt(-5.0)
        & metrics["minimum_domain_gain"].ge(0.0)
    )
    passing = metrics.loc[metrics["passes_source_gate"]].copy()
    if passing.empty:
        return {"risk": str(config["risks"][0]), "eta": 0.0, "source_gate": False}, metrics
    passing["robust_score"] = passing[
        ["gain", "worst_month_gain", "minimum_domain_gain"]
    ].min(axis=1)
    selected = passing.sort_values(
        ["robust_score", "gain", "eta"], ascending=[False, False, True]
    ).iloc[0]
    return {
        "risk": str(selected["risk"]),
        "eta": float(selected["eta"]),
        "source_gate": True,
    }, metrics


def run_transition(
    name: str,
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Repeat selection on source labels, freeze, and open one future audit."""

    corrections, crossfit = crossfit_source(source_frame, source_parent, config)
    recipe, source_metrics = _source_recipe(
        source_frame, source_parent, corrections, config
    )
    spec, selection = fit_stable_spec(
        build_features(source_frame),
        source_frame["target"].to_numpy(np.float64),
        source_parent,
        environment_labels(source_frame),
        str(recipe["risk"]),
        config,
    )
    correction = predict_correction(
        spec, build_features(audit_frame), float(config["correction_cap"])
    )
    candidate = np.clip(
        np.asarray(audit_parent) + float(recipe["eta"]) * correction,
        0.001,
        0.999,
    )
    result = diagnostics(
        audit_frame, audit_parent, candidate, np.ones(len(audit_frame), dtype=bool)
    )
    summary = {
        "axis": name,
        "source_rows": int(len(source_frame)),
        "audit_rows": int(len(audit_frame)),
        "selected_recipe": recipe,
        "selected_feature_count": int(len(spec.feature_names)),
        "selected_features": list(spec.feature_names),
        "selection": selection,
        "crossfit": crossfit,
        "audit": result,
    }
    return summary, candidate, source_metrics.assign(axis=name)


def _historical_frame(
    raw: pd.DataFrame, state_dir: Path, year: int
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = _add_domain_and_pressure(
        raw.loc[raw["season"].eq(int(year))].reset_index(drop=True)
    )
    with np.load(state_dir / f"selected_state_o{year}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
        checks = (
            np.array_equal(target, rows[TARGET].to_numpy(np.float64)),
            np.array_equal(saved["game_month"].astype(np.int16), rows["game_month"].to_numpy(np.int16)),
            np.array_equal(saved["pitcher_id"].astype(np.int64), rows["pitcher_id"].to_numpy(np.int64)),
            np.array_equal(saved["batter_id"].astype(np.int64), rows["batter_id"].to_numpy(np.int64)),
        )
        if not all(checks):
            raise ValueError(f"historical state parity failed for {year}: {checks}")
        rows["target"] = target
        rows["domain3"] = saved["domain3"].astype(str)
    return rows, parent


def _current_axes(
    project: Path, raw: pd.DataFrame, final_parent_dir: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    axes = _cached_v25_axes(project, raw)
    profiles = pd.read_csv(final_parent_dir / "trackman_profiles_2023_2025.csv")
    parents: dict[str, np.ndarray] = {}
    for name, frame in axes.items():
        season = int(frame["season"].iloc[0])
        eta15 = eta_parent(frame, CURRENT_ETA)
        parent, _ = apply_final_gate(
            frame,
            eta15,
            profiles.loc[profiles["season"].eq(season)],
            _source_global_rate(raw, name),
        )
        parents[name] = parent
    return axes, parents


def run(
    project: Path,
    state_dir: Path,
    final_parent_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    state_dir = state_dir.resolve()
    final_parent_dir = final_parent_dir.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    historical = {}
    historical_parent = {}
    for year in (2022, 2023, 2024):
        historical[year], historical_parent[year] = _historical_frame(raw, state_dir, year)
    axes, current_parent = _current_axes(project, raw, final_parent_dir)

    early22 = historical[2022]["game_month"].le(7).to_numpy()
    late22 = historical[2022]["game_month"].ge(8).to_numpy()
    full24 = axes["outer_full_2024"]
    early24 = full24["game_month"].le(7).to_numpy()
    transitions = {
        "early22_to_late22": (
            historical[2022].loc[early22].reset_index(drop=True),
            historical_parent[2022][early22],
            historical[2022].loc[late22].reset_index(drop=True),
            historical_parent[2022][late22],
        ),
        "full22_to_full23": (
            historical[2022],
            historical_parent[2022],
            historical[2023],
            historical_parent[2023],
        ),
        "full23_to_full24_historical": (
            historical[2023],
            historical_parent[2023],
            historical[2024],
            historical_parent[2024],
        ),
        "late23_to_full24_exact": (
            axes["selection_late_2023"],
            current_parent["selection_late_2023"],
            full24,
            current_parent["outer_full_2024"],
        ),
        "early24_to_late24_exact": (
            full24.loc[early24].reset_index(drop=True),
            current_parent["outer_full_2024"][early24],
            axes["replication_late_2024"],
            current_parent["replication_late_2024"],
        ),
    }
    summaries: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_frames = []
    for name, values in transitions.items():
        print(f"[v78] {name}", flush=True)
        summary, candidate, metrics = run_transition(name, *values, config)
        summaries[name] = summary
        predictions[name] = candidate
        metric_frames.append(metrics)

    primary_names = ("early22_to_late22", "full22_to_full23")
    primary = [summaries[name] for name in primary_names]
    gates = {
        "primary_source_selected_nonzero_eta": all(
            float(item["selected_recipe"]["eta"]) > 0.0 for item in primary
        ),
        "primary_gains_positive": all(float(item["audit"]["gain"]) > 0.0 for item in primary),
        "primary_month_fraction_at_least_075": all(
            float(item["audit"]["positive_month_fraction"]) >= 0.75 for item in primary
        ),
        "primary_worst_month_above_minus_5": all(
            float(item["audit"]["worst_month_gain"]) > -5.0 for item in primary
        ),
        "primary_domains_nonnegative": all(
            float(item["audit"]["minimum_domain_gain"]) >= 0.0 for item in primary
        ),
    }
    passes_primary = bool(all(gates.values()))
    compact_rows = []
    for name, item in summaries.items():
        compact_rows.append(
            {
                "axis": name,
                "selected_risk": item["selected_recipe"]["risk"],
                "selected_eta": item["selected_recipe"]["eta"],
                "selected_feature_count": item["selected_feature_count"],
                "gain": item["audit"]["gain"],
                "positive_month_fraction": item["audit"]["positive_month_fraction"],
                "worst_month_gain": item["audit"]["worst_month_gain"],
                "minimum_domain_gain": item["audit"]["minimum_domain_gain"],
                "mean_abs_shift": item["audit"]["mean_abs_shift"],
            }
        )
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "primary_axes": list(primary_names),
        "audits": summaries,
        "gates": gates,
        "passes_primary_mechanism_gate": passes_primary,
        "eligible_for_packaging": False,
        "packaging_reason": (
            "dependence-aware bootstrap and family Reality Check are still required"
            if passes_primary
            else "primary mechanism gate failed"
        ),
        "test_csv_read": False,
        "row_local_features_only": True,
        "test_aggregate_used": False,
        "current_pitch_physics_or_location_used": False,
    }
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output_dir / "source_trial_ledger.csv", index=False
    )
    pd.DataFrame(compact_rows).to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "metrics": compact_rows}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/configs/v78_environment_stable_residual.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.project,
        args.state_dir,
        args.final_parent_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
