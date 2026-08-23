"""Low-capacity temporal TrackMan profile residual models above public 1158.

This experiment uses the target-free origin profiles audited by v67.  A model
is fitted on pitcher-level residuals from one source period and transferred to
the next period.  All regularization and amplitude calibration use source-only
out-of-fold predictions.  Audit targets never choose alpha or blend weight.

The model is deliberately small and row-local at inference: one frozen
correction is looked up by the current row's pitcher ID and is applied only in
the already-established R_ANCHOR domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.v30_diverse_covariance_screen import _cached_v25_axes, diagnostics
from src.v67_trackman_command_proxy_census import pitcher_residual_table


ALPHAS = (1.0, 10.0, 100.0, 1000.0)
CORRECTION_CAP = 0.02
TRACKMAN_RELIABILITY_PRIOR = 500.0
DOMAIN = "R_ANCHOR"

CORE_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
FEATURE_SETS = {
    "single_ivb": ("tm_induced_vert_break_mean",),
    "means": tuple(
        [f"tm_{metric}_mean" for metric in CORE_METRICS]
        + [f"tm_latest_{metric}_mean" for metric in CORE_METRICS]
    ),
    "repeatability": tuple(
        [f"tm_{metric}_std" for metric in CORE_METRICS]
        + [f"tm_latest_{metric}_std" for metric in CORE_METRICS]
        + [f"tm_within_pitch_{metric}_std" for metric in CORE_METRICS]
    ),
    "physics_core": tuple(
        [f"tm_{metric}_mean" for metric in CORE_METRICS]
        + [f"tm_{metric}_std" for metric in CORE_METRICS]
        + [f"tm_latest_{metric}_mean" for metric in CORE_METRICS]
        + [f"tm_latest_{metric}_std" for metric in CORE_METRICS]
        + [f"tm_within_pitch_{metric}_std" for metric in CORE_METRICS]
    ),
}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_mse(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    return float(np.sum(weights * np.square(target - prediction)) / np.sum(weights))


def _fit_matrix(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, tuple[SimpleImputer, StandardScaler, Ridge, float]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(train_x))
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(transformed, train_y, sample_weight=train_weight)
    train_prediction = model.predict(transformed)
    center = _weighted_mean(train_prediction, train_weight)
    prediction = model.predict(scaler.transform(imputer.transform(predict_x))) - center
    return prediction.astype(np.float64), (imputer, scaler, model, center)


def fit_source_only_ridge(
    source: pd.DataFrame,
    audit: pd.DataFrame,
    features: tuple[str, ...],
    *,
    seed: int = 6801,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose ridge strength and correction amplitude from source OOF only."""

    missing = set(features) - set(source.columns)
    missing |= set(features) - set(audit.columns)
    if missing:
        raise ValueError(f"profile features missing: {sorted(missing)}")
    x = source[list(features)].to_numpy(np.float64)
    y = source["residual_eb"].to_numpy(np.float64)
    weight = source["reliability"].to_numpy(np.float64)
    audit_x = audit[list(features)].to_numpy(np.float64)
    splits = min(5, len(source))
    if splits < 3:
        raise ValueError("at least three source pitchers are required")
    fold = KFold(n_splits=splits, shuffle=True, random_state=seed)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for alpha in ALPHAS:
        oof = np.zeros(len(source), dtype=np.float64)
        for train_index, valid_index in fold.split(x):
            prediction, _ = _fit_matrix(
                x[train_index],
                y[train_index],
                weight[train_index],
                x[valid_index],
                alpha,
            )
            oof[valid_index] = prediction
        candidates.append((alpha, _weighted_mse(y, oof, weight), oof))
    selected_alpha, selected_mse, oof = min(candidates, key=lambda item: item[1])
    oof_variance = _weighted_mean(np.square(oof), weight)
    if oof_variance <= 1e-15:
        eta = 0.0
    else:
        eta = float(
            np.clip(_weighted_mean(y * oof, weight) / oof_variance, 0.0, 1.0)
        )
    raw_audit, fitted = _fit_matrix(x, y, weight, audit_x, selected_alpha)
    correction = eta * raw_audit
    diagnostic = {
        "source_pitchers": int(len(source)),
        "audit_pitchers": int(len(audit)),
        "feature_count": int(len(features)),
        "selected_alpha": float(selected_alpha),
        "source_oof_mse": float(selected_mse),
        "source_zero_mse": _weighted_mse(y, np.zeros_like(y), weight),
        "source_oof_eta": eta,
        "source_oof_correlation": float(np.corrcoef(y, oof)[0, 1])
        if np.std(oof) > 0 and np.std(y) > 0
        else None,
        "raw_audit_sd": float(np.std(raw_audit)),
    }
    del fitted
    return correction, diagnostic


def profile_training_table(
    frame: pd.DataFrame,
    parent: np.ndarray,
    profile: pd.DataFrame,
    domain: str = DOMAIN,
) -> pd.DataFrame:
    residual = pitcher_residual_table(frame, parent, domain)
    return residual.merge(profile, on="pitcher_id", how="inner", validate="one_to_one")


def apply_pitcher_correction(
    frame: pd.DataFrame,
    parent: np.ndarray,
    audit_profile: pd.DataFrame,
    pitcher_correction: np.ndarray,
    domain: str = DOMAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = audit_profile[["pitcher_id", "tm_n"]].copy()
    lookup["correction"] = np.asarray(pitcher_correction, dtype=np.float64)
    keys = frame[["pitcher_id"]].copy()
    keys["__row_order"] = np.arange(len(keys), dtype=np.int64)
    joined = keys.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    joined = joined.sort_values("__row_order", kind="stable")
    correction = joined["correction"].fillna(0.0).to_numpy(np.float64)
    tm_n = joined["tm_n"].fillna(0.0).to_numpy(np.float64)
    reliability = tm_n / (tm_n + TRACKMAN_RELIABILITY_PRIOR)
    correction = np.clip(
        correction * reliability, -CORRECTION_CAP, CORRECTION_CAP
    )
    domain_mask = (
        np.ones(len(frame), dtype=bool)
        if domain == "ALL"
        else frame["domain3"].astype(str).eq(domain).to_numpy()
    )
    active = domain_mask & joined["correction"].notna().to_numpy()
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + correction[active], 0.001, 0.999
    )
    return output, active, correction


def _transition(
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    source_profile: pd.DataFrame,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    audit_profile: pd.DataFrame,
    feature_name: str,
    domain: str = DOMAIN,
) -> tuple[dict[str, object], np.ndarray]:
    features = FEATURE_SETS[feature_name]
    source = profile_training_table(
        source_frame, source_parent, source_profile, domain=domain
    )
    correction, fit_audit = fit_source_only_ridge(source, audit_profile, features)
    candidate, active, row_correction = apply_pitcher_correction(
        audit_frame, audit_parent, audit_profile, correction, domain=domain
    )
    result = diagnostics(audit_frame, audit_parent, candidate, active)
    result.update(
        {
            **fit_audit,
            "active_rows": int(active.sum()),
            "active_fraction": float(active.mean()),
            "correction_sd": float(np.std(row_correction[active]))
            if active.any()
            else 0.0,
            "correction_max_abs": float(np.max(np.abs(row_correction[active])))
            if active.any()
            else 0.0,
        }
    )
    return result, candidate


def run(
    project: Path,
    profiles_path: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    profiles_path = profiles_path.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    profiles = pd.read_csv(profiles_path, low_memory=False)
    profile23 = profiles.loc[profiles["origin"].eq(2023)].drop(columns="origin")
    profile24 = profiles.loc[profiles["origin"].eq(2024)].drop(columns="origin")
    axes = _cached_v25_axes(project, raw)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    cache23 = np.load(current_oof_dir / "selection_late_2023.npz")
    cache24 = np.load(current_oof_dir / "outer_full_2024.npz")
    cache_rep = np.load(current_oof_dir / "replication_late_2024.npz")
    parent23 = cache23["final_gate_parent"].astype(np.float64)
    parent24 = cache24["final_gate_parent"].astype(np.float64)
    parent_rep = cache_rep["final_gate_parent"].astype(np.float64)
    if not np.array_equal(cache23["target"], late23["target"].to_numpy(np.float64)):
        raise ValueError("late-2023 parity failure")
    if not np.array_equal(cache24["target"], full24["target"].to_numpy(np.float64)):
        raise ValueError("full-2024 parity failure")
    if not np.array_equal(cache_rep["target"], replication24["target"].to_numpy(np.float64)):
        raise ValueError("late-2024 parity failure")

    early_mask = full24["game_month"].le(7).to_numpy()
    early24 = full24.loc[early_mask].reset_index(drop=True)
    parent_early24 = parent24[early_mask]
    rows: list[dict[str, object]] = []
    saved_predictions: dict[str, np.ndarray] = {}
    for feature_name in FEATURE_SETS:
        full_result, full_candidate = _transition(
            late23,
            parent23,
            profile23,
            full24,
            parent24,
            profile24,
            feature_name,
        )
        rows.append(
            {
                "feature_set": feature_name,
                "transition": "late23_to_full24",
                **full_result,
            }
        )
        replication_result, replication_candidate = _transition(
            early24,
            parent_early24,
            profile24,
            replication24,
            parent_rep,
            profile24,
            feature_name,
        )
        rows.append(
            {
                "feature_set": feature_name,
                "transition": "early24_to_late24",
                **replication_result,
            }
        )
        saved_predictions[f"{feature_name}_full24"] = full_candidate
        saved_predictions[f"{feature_name}_replication24"] = replication_candidate

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary_rows = (
        metrics.groupby("feature_set", observed=True)
        .agg(
            minimum_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            minimum_month_fraction=("positive_month_fraction", "min"),
            worst_month_gain=("worst_month_gain", "min"),
            maximum_abs_shift=("mean_abs_shift", "max"),
        )
        .reset_index()
        .sort_values(["minimum_gain", "mean_gain"], ascending=False)
    )
    summary_rows.to_csv(output_dir / "robust_summary.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **saved_predictions)
    eligible = summary_rows.loc[
        summary_rows["minimum_gain"].gt(0)
        & summary_rows["minimum_month_fraction"].ge(0.5)
        & summary_rows["worst_month_gain"].gt(-0.5)
    ]
    result = {
        "protocol": "V68_SOURCE_ONLY_TRACKMAN_PROFILE_RIDGE_ABOVE_1158_V1",
        "domain": DOMAIN,
        "source_only_hyperparameter_selection": True,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "summary": summary_rows.to_dict(orient="records"),
        "eligible_feature_sets": eligible["feature_set"].tolist(),
        "eligible_for_packaging": bool(len(eligible)),
        "decision": "research_only_even_if_positive_due_reused_2024_audit",
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.profiles, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
