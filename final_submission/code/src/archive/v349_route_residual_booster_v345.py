"""Honest route-specific residual boosters above the v345 analogue.

Three low-capacity, identity-free LightGBM regressors learn only the centred
parent residual.  Route eligibility is fixed on full-2022 -> late-2023.  New
models are then fit on late-2023 and the retained routes are opened once on
full-2024 above exact v345.  Hyperparameters, clipping and dose are fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V349_ROUTE_RESIDUAL_BOOSTER_V345_V1"
TARGET = "control_success"
ROUTES = ("F", "R_ANCHOR", "R_CORE")
CORRECTION_CAP = 0.02
PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 800,
    "max_bin": 63,
    "reg_alpha": 10.0,
    "reg_lambda": 100.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "objective": "regression_l2",
    "verbosity": -1,
    "n_jobs": 16,
    "random_state": 3490,
}
CATEGORICAL = [
    "game_month", "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "count_code", "platoon", "pressure_code",
]
DROP = ["row_id", "season", "pitcher_id", "batter_id", TARGET]


def route_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return {
        "F": frame["game_type"].astype(str).eq("F").to_numpy(),
        "R_ANCHOR": anchor,
        "R_CORE": regular & ~anchor,
    }


def build_features(frame: pd.DataFrame, parent: np.ndarray) -> pd.DataFrame:
    output = frame.drop(columns=DROP, errors="ignore").copy()
    output["parent_probability"] = np.asarray(parent, dtype=np.float64)
    output["parent_logit"] = np.log(
        np.clip(parent, 1e-5, 1 - 1e-5) / np.clip(1 - parent, 1e-5, 1 - 1e-5)
    )
    output["count_code"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    output["platoon"] = np.where(
        frame["pitcher_hand"].astype(str).eq(frame["batter_hand"].astype(str)),
        "SAME", "OPPOSITE",
    )
    output["pressure_code"] = np.select(
        [
            frame["num_runners_on"].gt(0) & frame["li"].ge(1.5),
            frame["num_runners_on"].gt(0),
            frame["li"].ge(1.5),
        ],
        ["RUNNER_HIGH", "RUNNER", "HIGH_LI"],
        default="QUIET",
    )
    for column in CATEGORICAL:
        if column in output:
            output[column] = output[column].fillna("__NA__").astype(str)
    for column in output.columns.difference(CATEGORICAL):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    categorical = [column for column in CATEGORICAL if column in features]
    numeric = [column for column in features if column not in categorical]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])
    return Pipeline([("pre", pre), ("model", LGBMRegressor(**PARAMS))])


def fit_predict_route(
    fit_frame: pd.DataFrame,
    fit_parent: np.ndarray,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    route: str,
) -> tuple[np.ndarray, Pipeline, dict[str, Any]]:
    fit_mask = route_masks(fit_frame)[route]
    audit_mask = route_masks(audit_frame)[route]
    fit_features = build_features(fit_frame.loc[fit_mask].reset_index(drop=True), fit_parent[fit_mask])
    audit_features = build_features(
        audit_frame.loc[audit_mask].reset_index(drop=True), audit_parent[audit_mask]
    ).reindex(columns=fit_features.columns)
    residual = (
        fit_frame.loc[fit_mask, TARGET].to_numpy(np.float64) - fit_parent[fit_mask]
    )
    residual_mean = float(residual.mean())
    model = make_pipeline(fit_features)
    model.fit(fit_features, residual - residual_mean)
    correction = np.clip(
        model.predict(audit_features).astype(np.float64),
        -CORRECTION_CAP,
        CORRECTION_CAP,
    )
    details = {
        "fit_rows": int(fit_mask.sum()),
        "audit_rows": int(audit_mask.sum()),
        "fit_residual_mean_removed": residual_mean,
        "correction_mean": float(correction.mean()),
        "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
        "correction_clip_fraction": float(np.mean(np.abs(correction) >= CORRECTION_CAP - 1e-15)),
    }
    return correction, model, details


def apply_routes(
    frame: pd.DataFrame,
    parent: np.ndarray,
    corrections: dict[str, np.ndarray],
    retained: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.zeros(len(frame), dtype=bool)
    masks = route_masks(frame)
    for route in ROUTES:
        if route not in retained:
            continue
        mask = masks[route]
        if int(mask.sum()) != len(corrections[route]):
            raise ValueError(f"correction alignment mismatch: {route}")
        output[mask] = np.clip(output[mask] + corrections[route], 0.001, 0.999)
        active |= mask
    return output, active


def metric_axes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate - parent))))


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64)
            for axis in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )

    source_corrections: dict[str, np.ndarray] = {}
    source_models: dict[str, Pipeline] = {}
    source_fit: dict[str, Any] = {}
    source_route_metrics: dict[str, Any] = {}
    source_masks = route_masks(frames["late_2023"])
    for route in ROUTES:
        correction, model, details = fit_predict_route(
            frames["full_2022"], parents["full_2022"],
            frames["late_2023"], parents["late_2023"], route,
        )
        source_corrections[route] = correction
        source_models[route] = model
        source_fit[route] = details
        route_candidate, route_active = apply_routes(
            frames["late_2023"], parents["late_2023"],
            source_corrections, {route},
        )
        source_route_metrics[route] = paired_metrics(
            metric_axes(frames["late_2023"]), parents["late_2023"],
            route_candidate, route_active,
        )
    retained = {
        route for route in ROUTES
        if source_route_metrics[route]["gain"] > 0.0
        and source_route_metrics[route]["positive_month_fraction"] >= 2.0 / 3.0
        and source_route_metrics[route]["worst_month_gain"] > -10.0
    }
    source_candidate, source_active = apply_routes(
        frames["late_2023"], parents["late_2023"], source_corrections, retained,
    )
    source_metrics = (
        {
            **paired_metrics(
                metric_axes(frames["late_2023"]), parents["late_2023"],
                source_candidate, source_active,
            ),
            "full_row_rms_shift": full_row_rms(
                parents["late_2023"], source_candidate
            ),
        }
        if retained
        else {
            "gain": 0.0,
            "active_gain": 0.0,
            "active_rows": 0,
            "positive_month_fraction": 0.0,
            "worst_month_gain": 0.0,
            "full_row_rms_shift": 0.0,
        }
    )
    if not retained or source_metrics["gain"] <= 0.0:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "retained_routes": sorted(retained),
            "source_route_metrics": source_route_metrics,
            "source_portfolio_metrics": source_metrics,
            "source_fit": source_fit,
            "candidate_gate_passed": False,
            "restrictions": {
                "official_train_only": True,
                "identity_columns_excluded": True,
                "route_selection_full2022_to_late2023_only": True,
                "full_2024_not_opened_after_source_failure": True,
                "test_csv_read": False,
                "public_score_used_for_selection": False,
                "row_local_inference": True,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    locked_corrections: dict[str, np.ndarray] = {}
    locked_fit: dict[str, Any] = {}
    for route in ROUTES:
        correction, model, details = fit_predict_route(
            frames["late_2023"], parents["late_2023"],
            frames["full_2024"], parents["full_2024"], route,
        )
        locked_corrections[route] = correction
        locked_fit[route] = details
        joblib.dump(model, output_dir / f"locked_{route.lower()}_residual_lgb.joblib", compress=3)
    locked_candidate, locked_active = apply_routes(
        frames["full_2024"], parents["full_2024"], locked_corrections, retained,
    )
    locked_metrics = paired_metrics(
        metric_axes(frames["full_2024"]), parents["full_2024"],
        locked_candidate, locked_active,
    )
    locked_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], locked_candidate
    )
    increment = locked_candidate - parents["full_2024"]
    nonzero = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = (
        float(np.corrcoef(increment[nonzero], v345_increment[nonzero])[0, 1])
        if int(nonzero.sum()) > 2
        and float(np.std(increment[nonzero])) > 0.0
        and float(np.std(v345_increment[nonzero])) > 0.0
        else 0.0
    )
    robustness = _robustness(
        metric_axes(frames["full_2024"]), parents["full_2024"],
        locked_candidate, locked_active, [parents["full_2024"], locked_candidate],
    )
    passed = bool(
        locked_metrics["gain"] >= 1.0
        and locked_metrics["positive_month_fraction"] >= 0.625
        and locked_metrics["worst_month_gain"] > -10.0
        and locked_metrics["full_row_rms_shift"] >= 0.001
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=source_candidate,
        active_late_2023=source_active,
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        direction_full_2024=increment,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "params": PARAMS,
        "correction_cap": CORRECTION_CAP,
        "retained_routes": sorted(retained),
        "source_route_metrics": source_route_metrics,
        "source_portfolio_metrics": source_metrics,
        "locked_full_2024": locked_metrics,
        "source_fit": source_fit,
        "locked_fit": locked_fit,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "candidate_gate_passed": passed,
        "restrictions": {
            "official_train_only": True,
            "identity_columns_excluded": True,
            "route_selection_full2022_to_late2023_only": True,
            "full_2024_opened_after_route_freeze": True,
            "hyperparameters_cap_and_dose_fixed": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v335_axes, args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
