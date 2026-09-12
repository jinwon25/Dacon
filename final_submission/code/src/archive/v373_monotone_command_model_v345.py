"""Strict-forward monotone command models as a diverse v345 complement."""

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


PROTOCOL = "V373_MONOTONE_COMMAND_MODEL_V345_V1"
TARGET = "control_success"
MODEL_FAMILIES = ("success_only", "success_failure")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.025, 0.05, 0.10)
POSITIVE_COLUMNS = {
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_success_rate",
}
NEGATIVE_COLUMNS = {
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_batter_middle_rate",
}
DROP = {"row_id", "season", TARGET}


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(frame)))
    category_columns = {"top_bottom", "game_type", "base_state"}
    for column in frame.columns:
        if column in DROP:
            continue
        if column in category_columns or frame[column].dtype == object:
            values = frame[column].astype("string").fillna("__NA__")
            categories = sorted(str(value) for value in values.unique())
            mapping = {value: index for index, value in enumerate(categories)}
            output[column] = values.astype(str).map(mapping).fillna(-1).astype(np.float32)
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.float32)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.float32)
    output["eng_count_code"] = balls * 3.0 + strikes
    output["eng_balls_minus_strikes"] = balls - strikes
    output["eng_three_ball"] = (balls >= 3).astype(np.float32)
    output["eng_two_strike"] = (strikes >= 2).astype(np.float32)
    output["eng_hand_match"] = (
        pd.to_numeric(frame["pitcher_hand"], errors="coerce").to_numpy()
        == pd.to_numeric(frame["batter_hand"], errors="coerce").to_numpy()
    ).astype(np.float32)
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        value = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(np.float32)
        output[f"eng_log1p_{column}"] = np.log1p(np.maximum(value, 0.0))
    p1 = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"], errors="coerce")
    p3 = pd.to_numeric(frame["asof_pitcher_prev3_game_success_rate"], errors="coerce")
    p5 = pd.to_numeric(frame["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    career = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
    output["eng_recent_success_delta"] = (0.5 * p1 + 0.3 * p3 + 0.2 * p5 - career).astype(np.float32)
    output["eng_recent_success_slope"] = (p1 - p5).astype(np.float32)
    m1 = pd.to_numeric(frame["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
    m5 = pd.to_numeric(frame["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    output["eng_recent_middle_slope"] = (m1 - m5).astype(np.float32)
    mix = frame[[
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    mix = np.clip(mix, 1e-8, 1.0)
    output["eng_pitchmix_entropy"] = (-np.sum(mix * np.log(mix), axis=1)).astype(np.float32)
    if output.columns.duplicated().any():
        raise ValueError("duplicate feature names")
    return output


def constraints(columns: list[str], family: str) -> list[int]:
    result = []
    for column in columns:
        if column in POSITIVE_COLUMNS:
            result.append(1)
        elif family == "success_failure" and column in NEGATIVE_COLUMNS:
            result.append(-1)
        else:
            result.append(0)
    return result


def fit_predict(
    fit_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    family: str,
    seed: int,
) -> tuple[np.ndarray, lgb.LGBMRegressor, dict[str, Any]]:
    fit_x = build_features(fit_frame)
    audit_x = build_features(audit_frame).reindex(columns=fit_x.columns)
    newest = int(fit_frame["season"].max())
    sample_weight = np.where(
        fit_frame["season"].eq(newest).to_numpy(), 1.0, 0.60
    ).astype(np.float64)
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=700,
        max_bin=127,
        reg_alpha=5.0,
        reg_lambda=40.0,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        monotone_constraints=constraints(list(fit_x.columns), family),
        monotone_constraints_method="advanced",
        verbosity=-1,
        n_jobs=16,
        random_state=seed,
    )
    model.fit(fit_x, fit_frame[TARGET].to_numpy(np.float64), sample_weight=sample_weight)
    prediction = np.clip(model.predict(audit_x), 0.001, 0.999).astype(np.float64)
    details = {
        "fit_rows": len(fit_frame), "audit_rows": len(audit_frame),
        "feature_count": len(fit_x.columns),
        "positive_constraints": int(sum(value == 1 for value in model.monotone_constraints)),
        "negative_constraints": int(sum(value == -1 for value in model.monotone_constraints)),
        "prediction_mean": float(prediction.mean()),
        "prediction_sd": float(prediction.std()),
    }
    del fit_x, audit_x
    gc.collect()
    return prediction, model, details


def route_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return {
        "ALL": np.ones(len(frame), dtype=bool),
        "R_CORE": regular & ~anchor,
        "R_ANCHOR": anchor,
        "F": frame["game_type"].astype(str).eq("F").to_numpy(),
    }


def candidate(parent: np.ndarray, raw: np.ndarray, active: np.ndarray, weight: float) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * (raw[active] - output[active]), 0.001, 0.999
    )
    return output


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
    late23_mask = frames["full_2023"]["game_month"].ge(8).to_numpy()
    raw: dict[str, dict[str, np.ndarray]] = {}
    fit_details: dict[str, Any] = {}
    for index, family in enumerate(MODEL_FAMILIES):
        raw22, _, detail22 = fit_predict(
            train.loc[train["season"].between(2020, 2021)].reset_index(drop=True),
            frames["full_2022"], family, 3730 + index,
        )
        raw23_full, _, detail23 = fit_predict(
            train.loc[train["season"].between(2021, 2022)].reset_index(drop=True),
            frames["full_2023"], family, 3740 + index,
        )
        raw[family] = {
            "full_2022": raw22,
            "late_2023": raw23_full[late23_mask],
        }
        fit_details[family] = {"full_2022": detail22, "late_2023": detail23}

    trials: list[dict[str, Any]] = []
    payload: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    for family in MODEL_FAMILIES:
        for route in ROUTES:
            for weight in WEIGHTS:
                key = (family, route, weight)
                payload[key] = {}
                row: dict[str, Any] = {"family": family, "route": route, "weight": weight}
                for axis in ("full_2022", "late_2023"):
                    active = route_masks(frames[axis])[route]
                    value = candidate(parents[axis], raw[family][axis], active, weight)
                    payload[key][axis] = value
                    metrics = axis_metrics(frames[axis], parents[axis], value, active)
                    row[axis] = metrics
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
    passing = [row for row in trials if row["source_pass"]]
    passing.sort(
        key=lambda row: (
            row["minimum_source_gain"],
            row["full_2022"]["gain"] + row["late_2023"]["gain"],
        ), reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "two_fixed_monotonicity_schemas": True,
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
            "source_trials": trials, "fit_details": fit_details,
            "locked_origin_opened": False, "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    family, route, weight = str(chosen["family"]), str(chosen["route"]), float(chosen["weight"])
    raw24, model24, detail24 = fit_predict(
        train.loc[train["season"].between(2022, 2023)].reset_index(drop=True),
        frames["full_2024"], family, 3750 + MODEL_FAMILIES.index(family),
    )
    active24 = route_masks(frames["full_2024"])[route]
    candidate24 = candidate(parents["full_2024"], raw24, active24, weight)
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
    passed = bool(
        locked["gain"] >= 2.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
        and robustness["chronological_block"]["p05"] > 0.0
    )
    joblib.dump(
        {"model": model24, "family": family, "route": route, "weight": weight},
        output_dir / "locked_model.joblib", compress=3,
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[(family, route, weight)]["full_2022"],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[(family, route, weight)]["late_2023"],
        parent_full_2024=parents["full_2024"], candidate_full_2024=candidate24,
        raw_full_2024=raw24, active_full_2024=active24,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if passed else "locked_reject",
        "source_trials": trials, "selected_source_recipe": chosen,
        "fit_details": {**fit_details, "locked_full_2024": detail24},
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed, "restrictions": restrictions,
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
