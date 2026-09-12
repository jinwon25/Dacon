"""Strict-forward support-aware shrinkage toward an ID-free population model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms
from src.archive.v373_monotone_command_model_v345 import build_features


PROTOCOL = "V377_IDFREE_SUPPORT_SHRINK_V345_V1"
TARGET = "control_success"
SUPPORT_ROUTES = ("LOW", "DEVELOPING", "LOW_DEVELOPING", "ESTABLISHED", "R_CORE")
WEIGHTS = (0.025, 0.05, 0.10)
IDENTITY_TOKENS = ("row_id", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id")


def idfree_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = build_features(frame)
    drop = [
        column for column in output.columns
        if column in IDENTITY_TOKENS
        or column.endswith("_id")
        or "team_id" in column
    ]
    return output.drop(columns=drop, errors="ignore")


def fit_predict(
    fit_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, lgb.LGBMClassifier, dict[str, Any]]:
    fit_x = idfree_features(fit_frame)
    audit_x = idfree_features(audit_frame).reindex(columns=fit_x.columns)
    newest = int(fit_frame["season"].max())
    sample_weight = np.where(
        fit_frame["season"].eq(newest).to_numpy(), 1.0, 0.60
    ).astype(np.float64)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=1000,
        max_bin=127,
        reg_alpha=8.0,
        reg_lambda=60.0,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        verbosity=-1,
        n_jobs=16,
        random_state=seed,
    )
    model.fit(
        fit_x, fit_frame[TARGET].to_numpy(np.int8), sample_weight=sample_weight
    )
    prediction = np.clip(model.predict_proba(audit_x)[:, 1], 0.001, 0.999)
    details = {
        "fit_rows": len(fit_frame), "audit_rows": len(audit_frame),
        "feature_count": len(fit_x.columns),
        "prediction_mean": float(prediction.mean()),
        "prediction_sd": float(prediction.std()),
    }
    del fit_x, audit_x
    gc.collect()
    return prediction, model, details


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    core = regular & ~anchor
    support = pd.to_numeric(
        frame["asof_pitcher_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    return {
        "LOW": core & (support < 100.0),
        "DEVELOPING": core & (support >= 100.0) & (support < 800.0),
        "LOW_DEVELOPING": core & (support < 800.0),
        "ESTABLISHED": core & (support >= 800.0),
        "R_CORE": core,
    }


def candidate(
    parent: np.ndarray, population: np.ndarray, active: np.ndarray, weight: float
) -> np.ndarray:
    out = np.asarray(parent, dtype=np.float64).copy()
    out[active] = np.clip(
        out[active] + float(weight) * (population[active] - out[active]),
        0.001, 0.999,
    )
    return out


def run(train_csv: Path, v335_axes: Path, v345_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "full_2023": train.loc[train["season"].eq(2023)].reset_index(drop=True),
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
        v345_increment = saved["candidate_full_2024"].astype(np.float64) - saved[
            "parent_full_2024"
        ].astype(np.float64)
    population22, _, detail22 = fit_predict(
        train.loc[train["season"].between(2020, 2021)].reset_index(drop=True),
        frames["full_2022"], 3771,
    )
    population23, _, detail23 = fit_predict(
        train.loc[train["season"].between(2021, 2022)].reset_index(drop=True),
        frames["full_2023"], 3772,
    )
    late23 = frames["full_2023"]["game_month"].ge(8).to_numpy()
    population = {"full_2022": population22, "late_2023": population23[late23]}
    trials: list[dict[str, Any]] = []
    payload: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for route in SUPPORT_ROUTES:
        for weight in WEIGHTS:
            key = (route, weight)
            payload[key] = {}
            row: dict[str, Any] = {"support_route": route, "weight": weight}
            for axis in ("full_2022", "late_2023"):
                active = masks(frames[axis])[route]
                value = candidate(parents[axis], population[axis], active, weight)
                payload[key][axis] = value
                row[axis] = axis_metrics(frames[axis], parents[axis], value, active)
            row["minimum_source_gain"] = min(
                row["full_2022"]["gain"], row["late_2023"]["gain"]
            )
            row["source_pass"] = bool(
                row["full_2022"]["gain"] > 0.0
                and row["late_2023"]["gain"] > 0.0
                and row["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
                and row["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
                and row["full_2022"]["worst_month_gain"] > -10.0
                and row["late_2023"]["worst_month_gain"] > -10.0
            )
            trials.append(row)
    passing = sorted(
        (row for row in trials if row["source_pass"]),
        key=lambda row: (row["minimum_source_gain"], row["full_2022"]["gain"] + row["late_2023"]["gain"]),
        reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "population_model_has_no_player_or_team_identifiers": True,
        "fixed_support_routes_and_low_blend_doses": True,
        "source_selection_full2022_and_late2023_only": True,
        "full2024_model_fit_only_after_source_gate": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "source_trials": trials,
            "fit_details": {"full_2022": detail22, "late_2023": detail23},
            "locked_origin_opened": False, "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    route, weight = str(chosen["support_route"]), float(chosen["weight"])
    population24, model24, detail24 = fit_predict(
        train.loc[train["season"].between(2022, 2023)].reset_index(drop=True),
        frames["full_2024"], 3773,
    )
    active24 = masks(frames["full_2024"])[route]
    candidate24 = candidate(parents["full_2024"], population24, active24, weight)
    locked = axis_metrics(frames["full_2024"], parents["full_2024"], candidate24, active24)
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], candidate24)
    increment = candidate24 - parents["full_2024"]
    union = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(increment[union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidate24, active24,
        [parents["full_2024"], candidate24],
    )
    eligible = bool(
        locked["gain"] >= 2.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
        and robustness["chronological_block"]["p05"] > 0.0
    )
    joblib.dump(
        {"model": model24, "support_route": route, "weight": weight},
        output_dir / "locked_model.joblib", compress=3,
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[(route, weight)]["full_2022"],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[(route, weight)]["late_2023"],
        parent_full_2024=parents["full_2024"], candidate_full_2024=candidate24,
        population_full_2024=population24, active_full_2024=active24,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if eligible else "locked_reject",
        "source_trials": trials, "selected_source_recipe": chosen,
        "fit_details": {
            "full_2022": detail22, "late_2023": detail23, "full_2024": detail24,
        },
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": eligible, "restrictions": restrictions,
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
        args.train_csv, args.v335_axes, args.v345_axes, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
