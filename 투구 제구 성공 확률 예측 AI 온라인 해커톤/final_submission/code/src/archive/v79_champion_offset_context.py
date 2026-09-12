"""Preregistered low-DOF context GLM with the champion logit as a fixed offset.

The champion owns season-specific baseline risk and player state.  This model
learns only source-period relative effects of count, handedness, base/out
state, leverage and the row-local command composition.  Corrections are
centred by source domain and applied independently to every audit row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

from src.core.diagnostics import diagnostics
from src.archive.v78_environment_stable_residual import _current_axes, _historical_frame


PROTOCOL = "V79_CHAMPION_OFFSET_CONTEXT_GLM_V1"
TARGET = "control_success"
EPSILON = 1e-5


@dataclass
class OffsetModel:
    variant: str
    numeric_names: tuple[str, ...]
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    categorical_names: tuple[str, ...]
    encoder: OneHotEncoder | None
    coefficients: np.ndarray
    correction_cap: float
    domain_means: dict[str, float]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return np.where(np.isfinite(values), values, float(default))


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 0.005, 0.995)
    return np.log(value) - np.log1p(-value)


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed ID-free continuous feature bank."""

    balls = _numeric(frame, "balls_before")
    strikes = _numeric(frame, "strikes_before")
    p_n = np.maximum(_numeric(frame, "asof_pitcher_n"), 0.0)
    b_n = np.maximum(_numeric(frame, "asof_batter_n"), 0.0)
    p_success = np.clip(_numeric(frame, "asof_pitcher_success_rate", 0.5), 0.0, 1.0)
    b_success = np.clip(_numeric(frame, "asof_batter_success_rate", 0.5), 0.0, 1.0)
    p_middle = np.clip(_numeric(frame, "asof_pitcher_middle_rate"), 0.0, 1.0)
    p_reverse = np.clip(_numeric(frame, "asof_pitcher_reverse_rate"), 0.0, 1.0)
    p_wayoff = np.clip(1.0 - p_success - p_middle - p_reverse, 0.0, 1.0)
    b_middle = np.clip(_numeric(frame, "asof_batter_middle_rate"), 0.0, 1.0)
    prev1 = _numeric(frame, "asof_pitcher_prev1_game_success_rate", 0.5)
    prev3 = _numeric(frame, "asof_pitcher_prev3_game_success_rate", 0.5)
    prev5 = _numeric(frame, "asof_pitcher_prev5_game_success_rate", 0.5)
    middle1 = _numeric(frame, "asof_pitcher_prev1_game_middle_rate")
    middle3 = _numeric(frame, "asof_pitcher_prev3_game_middle_rate")
    middle5 = _numeric(frame, "asof_pitcher_prev5_game_middle_rate")
    mix = np.column_stack(
        [
            _numeric(frame, "asof_pitcher_fastball_rate", 1.0 / 3.0),
            _numeric(frame, "asof_pitcher_breaking_rate", 1.0 / 3.0),
            _numeric(frame, "asof_pitcher_offspeed_rate", 1.0 / 3.0),
        ]
    )
    mix = np.clip(np.where(np.isfinite(mix), mix, 1.0 / 3.0), 1e-6, None)
    mix /= mix.sum(axis=1, keepdims=True)
    pitcher_hand = _safe_string(frame["pitcher_hand"])
    batter_hand = _safe_string(frame["batter_hand"])

    output = pd.DataFrame(
        {
            "ctx_balls": balls / 3.0,
            "ctx_strikes": strikes / 2.0,
            "ctx_outs": _numeric(frame, "outs_before") / 2.0,
            "ctx_runners": _numeric(frame, "num_runners_on") / 3.0,
            "ctx_inning": np.clip(_numeric(frame, "inning"), 1.0, 12.0) / 9.0,
            "ctx_abs_score": np.minimum(
                np.abs(_numeric(frame, "score_diff_pitcher_team")), 10.0
            )
            / 10.0,
            "ctx_log_li": np.log1p(np.maximum(_numeric(frame, "li"), 0.0)),
            "ctx_win_gap": _numeric(frame, "home_win_expectancy", 0.5)
            - _numeric(frame, "away_win_expectancy", 0.5),
            "ctx_same_hand": (
                pitcher_hand.to_numpy() == batter_hand.to_numpy()
            ).astype(np.float64),
            "ctx_pressure": ((balls == 3.0) | (strikes == 2.0)).astype(np.float64),
            "cmp_pitcher_success_logit": _safe_logit(p_success),
            "cmp_batter_success_logit": _safe_logit(b_success),
            "cmp_middle_logit": _safe_logit(p_middle),
            "cmp_reverse_logit": _safe_logit(p_reverse),
            "cmp_wayoff_logit": _safe_logit(p_wayoff),
            "cmp_batter_middle_logit": _safe_logit(b_middle),
            "cmp_success_recent_delta": 0.5 * prev1 + 0.3 * prev3 + 0.2 * prev5 - p_success,
            "cmp_middle_recent_delta": 0.5 * middle1 + 0.3 * middle3 + 0.2 * middle5 - p_middle,
            "cmp_success_curvature": prev1 - 2.0 * prev3 + prev5,
            "cmp_middle_curvature": middle1 - 2.0 * middle3 + middle5,
            "cmp_pitcher_reliability": p_n / (p_n + 160.0),
            "cmp_batter_reliability": b_n / (b_n + 160.0),
            "cmp_log_pitcher_n": np.log1p(p_n),
            "cmp_log_batter_n": np.log1p(b_n),
            "cmp_strike_ball_margin": _numeric(frame, "asof_pitcher_strike_rate")
            - _numeric(frame, "asof_pitcher_ball_rate"),
            "cmp_pitchmix_entropy": -np.sum(mix * np.log(mix), axis=1),
        }
    )
    if not np.isfinite(output.to_numpy(np.float64)).all():
        raise ValueError("non-finite v79 numeric feature")
    return output


def categorical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return low-cardinality baseball interactions without entity IDs."""

    balls = _safe_string(frame["balls_before"])
    strikes = _safe_string(frame["strikes_before"])
    count = balls + "-" + strikes
    pitcher_hand = _safe_string(frame["pitcher_hand"])
    batter_hand = _safe_string(frame["batter_hand"])
    domain = _safe_string(frame["domain3"])
    base = _safe_string(frame["base_state"])
    outs = _safe_string(frame["outs_before"])
    top_bottom = _safe_string(frame["top_bottom"])
    inning = pd.cut(
        _numeric(frame, "inning"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string").fillna("__MISSING__").astype(str)
    score = pd.cut(
        _numeric(frame, "score_diff_pitcher_team"),
        bins=(-np.inf, -3, -1, 1, 3, np.inf),
        labels=("behind4", "behind", "close", "ahead", "ahead4"),
    ).astype("string").fillna("__MISSING__").astype(str)
    leverage = pd.cut(
        _numeric(frame, "li"),
        bins=(-np.inf, 0.75, 1.5, 3.0, np.inf),
        labels=("low", "medium", "high", "very_high"),
    ).astype("string").fillna("__MISSING__").astype(str)
    return pd.DataFrame(
        {
            "count": count,
            "count_hands": count + "|" + pitcher_hand + "|" + batter_hand,
            "base_out": base + "|" + outs,
            "inning_side": inning + "|" + top_bottom,
            "score_leverage": score + "|" + leverage,
            "domain_count": domain + "|" + count,
            "domain_base_out": domain + "|" + base + "|" + outs,
        }
    )


def _variant_columns(variant: str, numeric: pd.DataFrame) -> tuple[list[str], bool]:
    if variant == "context":
        return [name for name in numeric if name.startswith("ctx_")], True
    if variant == "composition":
        return [name for name in numeric if name.startswith("cmp_")], False
    if variant == "combined":
        return list(numeric.columns), True
    raise ValueError(f"unknown v79 variant: {variant}")


def _design_fit(
    frame: pd.DataFrame, variant: str
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    numeric = numeric_features(frame)
    numeric_names, use_category = _variant_columns(variant, numeric)
    values = numeric[numeric_names].to_numpy(np.float64)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    parts: list[sparse.spmatrix] = [sparse.csr_matrix((values - means) / scales)]
    encoder: OneHotEncoder | None = None
    category_names: tuple[str, ...] = ()
    if use_category:
        categories = categorical_features(frame)
        category_names = tuple(categories.columns)
        encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
        parts.append(encoder.fit_transform(categories))
    matrix = sparse.hstack(parts, format="csr", dtype=np.float64)
    state = {
        "numeric_names": tuple(numeric_names),
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_names": category_names,
        "encoder": encoder,
    }
    return matrix, state


def _design_transform(frame: pd.DataFrame, state: dict[str, Any]) -> sparse.csr_matrix:
    numeric = numeric_features(frame)
    names = list(state["numeric_names"])
    values = numeric[names].to_numpy(np.float64)
    parts: list[sparse.spmatrix] = [
        sparse.csr_matrix(
            (values - state["numeric_means"]) / state["numeric_scales"]
        )
    ]
    encoder = state["encoder"]
    if encoder is not None:
        categories = categorical_features(frame).loc[
            :, list(state["categorical_names"])
        ]
        parts.append(encoder.transform(categories))
    return sparse.hstack(parts, format="csr", dtype=np.float64)


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_coefficients(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    offset: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a regularized offset logistic model without a free intercept."""

    y = np.asarray(target, dtype=np.float64)
    base = _safe_logit(offset)
    weight = np.asarray(sample_weight, dtype=np.float64)
    weight = weight / weight.mean()
    if y.shape != base.shape or y.shape != weight.shape or len(y) != matrix.shape[0]:
        raise ValueError("v79 fit arrays are not aligned")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        probability = _expit(base + matrix @ beta)
        nll = -np.mean(
            weight
            * (y * np.log(np.clip(probability, EPSILON, 1.0))
               + (1.0 - y) * np.log(np.clip(1.0 - probability, EPSILON, 1.0)))
        )
        penalty = 0.5 * float(l2) * float(beta @ beta)
        gradient = np.asarray(
            matrix.T @ (weight * (probability - y)) / len(y)
        ).reshape(-1)
        gradient += float(l2) * beta
        return float(nll + penalty), gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maximum_iterations), "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success and int(result.nit) < int(maximum_iterations):
        raise RuntimeError(f"v79 offset fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "feature_count": int(matrix.shape[1]),
    }


def _environment_weights(frame: pd.DataFrame) -> np.ndarray:
    month = _numeric(frame, "game_month").astype(np.int16)
    band = np.where(month <= 5, "early", np.where(month <= 7, "mid", "late"))
    environment = frame["domain3"].astype(str).to_numpy() + "|" + band
    unique, counts = np.unique(environment, return_counts=True)
    inverse = {name: 1.0 / count for name, count in zip(unique, counts, strict=True)}
    weight = np.asarray([inverse[name] for name in environment], dtype=np.float64)
    return weight / weight.mean()


def _domain_correction_means(
    frame: pd.DataFrame, correction: np.ndarray
) -> dict[str, float]:
    domain = frame["domain3"].astype(str).to_numpy()
    return {
        name: float(np.mean(correction[domain == name])) for name in np.unique(domain)
    }


def fit_offset_model(
    frame: pd.DataFrame,
    parent: np.ndarray,
    variant: str,
    config: dict[str, Any],
) -> tuple[OffsetModel, dict[str, Any]]:
    matrix, state = _design_fit(frame, variant)
    coefficients, audit = fit_coefficients(
        matrix,
        frame["target"].to_numpy(np.float64),
        parent,
        _environment_weights(frame),
        l2=float(config["ridge_l2"]),
        maximum_iterations=int(config["maximum_iterations"]),
    )
    raw = np.clip(
        np.asarray(matrix @ coefficients).reshape(-1),
        -float(config["correction_cap_logit"]),
        float(config["correction_cap_logit"]),
    )
    model = OffsetModel(
        variant=variant,
        numeric_names=state["numeric_names"],
        numeric_means=state["numeric_means"],
        numeric_scales=state["numeric_scales"],
        categorical_names=state["categorical_names"],
        encoder=state["encoder"],
        coefficients=coefficients,
        correction_cap=float(config["correction_cap_logit"]),
        domain_means=_domain_correction_means(frame, raw),
    )
    audit["variant"] = variant
    audit["source_domain_means"] = model.domain_means
    return model, audit


def predict_correction(model: OffsetModel, frame: pd.DataFrame) -> np.ndarray:
    state = {
        "numeric_names": model.numeric_names,
        "numeric_means": model.numeric_means,
        "numeric_scales": model.numeric_scales,
        "categorical_names": model.categorical_names,
        "encoder": model.encoder,
    }
    matrix = _design_transform(frame, state)
    raw = np.clip(
        np.asarray(matrix @ model.coefficients).reshape(-1),
        -model.correction_cap,
        model.correction_cap,
    )
    domain = frame["domain3"].astype(str).to_numpy()
    centre = np.asarray([model.domain_means.get(name, 0.0) for name in domain])
    return np.clip(raw - centre, -model.correction_cap, model.correction_cap)


def rolling_month_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    month = _numeric(frame, "game_month").astype(np.int16)
    splits = []
    for train_end, valid_start, valid_end in ((5, 6, 7), (7, 8, 12)):
        train = month <= train_end
        valid = (month >= valid_start) & (month <= valid_end)
        if int(train.sum()) >= 1000 and int(valid.sum()) >= 1000:
            splits.append((train, valid))
    return splits


def source_crossfit(
    frame: pd.DataFrame,
    parent: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    corrections = {
        str(variant): np.zeros(len(frame), dtype=np.float64)
        for variant in config["variants"]
    }
    covered = np.zeros(len(frame), dtype=bool)
    fold_audits: list[dict[str, Any]] = []
    for train, valid in rolling_month_splits(frame):
        covered |= valid
        train_frame = frame.loc[train].reset_index(drop=True)
        valid_frame = frame.loc[valid].reset_index(drop=True)
        fold = {
            "train_max_month": int(frame.loc[train, "game_month"].max()),
            "valid_min_month": int(frame.loc[valid, "game_month"].min()),
            "train_rows": int(train.sum()),
            "valid_rows": int(valid.sum()),
            "models": {},
        }
        for variant in config["variants"]:
            model, audit = fit_offset_model(
                train_frame, np.asarray(parent)[train], str(variant), config
            )
            corrections[str(variant)][valid] = predict_correction(model, valid_frame)
            fold["models"][str(variant)] = audit
        fold_audits.append(fold)
    return corrections, covered, {"folds": fold_audits, "covered_rows": int(covered.sum())}


def select_source_recipe(
    frame: pd.DataFrame,
    parent: np.ndarray,
    corrections: dict[str, np.ndarray],
    covered: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    source = frame.loc[covered].reset_index(drop=True)
    source_parent = np.asarray(parent)[covered]
    for variant in config["variants"]:
        correction = corrections[str(variant)][covered]
        for eta in config["eta_grid"]:
            candidate = _expit(_safe_logit(source_parent) + float(eta) * correction)
            result = diagnostics(
                source,
                source_parent,
                candidate,
                np.ones(len(source), dtype=bool),
            )
            rows.append(
                {
                    "variant": str(variant),
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
        return {"variant": str(config["variants"][0]), "eta": 0.0, "source_gate": False}, metrics
    passing["robust_score"] = passing[
        ["gain", "worst_month_gain", "minimum_domain_gain"]
    ].min(axis=1)
    selected = passing.sort_values(
        ["robust_score", "gain", "eta"], ascending=[False, False, True]
    ).iloc[0]
    return {
        "variant": str(selected["variant"]),
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
    corrections, covered, crossfit = source_crossfit(
        source_frame, source_parent, config
    )
    if not covered.any():
        recipe = {"variant": str(config["variants"][0]), "eta": 0.0, "source_gate": False}
        source_metrics = pd.DataFrame()
    else:
        recipe, source_metrics = select_source_recipe(
            source_frame, source_parent, corrections, covered, config
        )
    model, fit_audit = fit_offset_model(
        source_frame, source_parent, str(recipe["variant"]), config
    )
    correction = predict_correction(model, audit_frame)
    candidate = _expit(
        _safe_logit(audit_parent) + float(recipe["eta"]) * correction
    )
    result = diagnostics(
        audit_frame,
        audit_parent,
        candidate,
        np.ones(len(audit_frame), dtype=bool),
    )
    return {
        "axis": name,
        "source_rows": int(len(source_frame)),
        "audit_rows": int(len(audit_frame)),
        "selected_recipe": recipe,
        "crossfit": crossfit,
        "final_fit": fit_audit,
        "audit": result,
    }, candidate, source_metrics.assign(axis=name)


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
    historical: dict[int, pd.DataFrame] = {}
    historical_parent: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        historical[year], historical_parent[year] = _historical_frame(
            raw, state_dir, year
        )
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
    ledgers = []
    compact = []
    for name, values in transitions.items():
        print(f"[v79] {name}", flush=True)
        summary, candidate, ledger = run_transition(name, *values, config)
        summaries[name] = summary
        predictions[name] = candidate
        if not ledger.empty:
            ledgers.append(ledger)
        audit = summary["audit"]
        compact.append(
            {
                "axis": name,
                "selected_variant": summary["selected_recipe"]["variant"],
                "selected_eta": summary["selected_recipe"]["eta"],
                "gain": audit["gain"],
                "positive_month_fraction": audit["positive_month_fraction"],
                "worst_month_gain": audit["worst_month_gain"],
                "minimum_domain_gain": audit["minimum_domain_gain"],
                "mean_abs_shift": audit["mean_abs_shift"],
            }
        )
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
    passes = bool(all(gates.values()))
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "primary_axes": list(primary_names),
        "audits": summaries,
        "gates": gates,
        "passes_primary_mechanism_gate": passes,
        "eligible_for_packaging": False,
        "packaging_reason": (
            "dependence-aware bootstrap and family Reality Check are still required"
            if passes
            else "primary mechanism gate failed"
        ),
        "test_csv_read": False,
        "row_local_features_only": True,
        "test_aggregate_used": False,
        "current_pitch_physics_or_location_used": False,
    }
    pd.concat(ledgers, ignore_index=True).to_csv(
        output_dir / "source_trial_ledger.csv", index=False
    )
    pd.DataFrame(compact).to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "metrics": compact}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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
