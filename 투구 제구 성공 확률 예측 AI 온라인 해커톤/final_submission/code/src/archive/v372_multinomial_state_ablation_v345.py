"""Source-only ablation of v371 multinomial current-season state families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms
from src.archive.v371_multinomial_season_state_v345 import (
    RATE_COLUMNS,
    TARGET,
    analytic_dose,
    apply,
    correction,
    fit_model,
    rcore,
    state_features,
)


PROTOCOL = "V372_MULTINOMIAL_STATE_ABLATION_V345_V1"
FAMILIES = (
    "success",
    "location",
    "called",
    "failure_all",
    "innovations",
    "all",
)


def family_columns(features: pd.DataFrame, family: str) -> list[str]:
    meta = ["season_log_n", "season_reliability", "returning_pitcher"]
    columns = list(features.columns)
    if family == "success":
        selected = [name for name in columns if "state_success_" in name]
    elif family == "location":
        selected = [
            name for name in columns
            if any(token in name for token in ("state_reverse_", "state_middle_", "bad_location"))
        ]
    elif family == "called":
        selected = [
            name for name in columns
            if any(token in name for token in ("state_ball_", "state_strike_", "called_result"))
        ]
    elif family == "failure_all":
        selected = [name for name in columns if name.startswith("state_") and "success" not in name]
    elif family == "innovations":
        selected = [name for name in columns if "innovation" in name]
    elif family == "all":
        selected = [name for name in columns if name.startswith("state_")]
    else:
        raise ValueError(f"unknown family: {family}")
    return list(dict.fromkeys(meta + selected))


def run(train_csv: Path, v335_axes: Path, v345_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        "asof_pitcher_n", *RATE_COLUMNS, TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
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
        v345_increment = saved["candidate_full_2024"].astype(np.float64) - saved[
            "parent_full_2024"
        ].astype(np.float64)
    features = {
        axis: state_features(train, frame, year)
        for axis, frame, year in (
            ("full_2022", frames["full_2022"], 2022),
            ("late_2023", frames["late_2023"], 2023),
            ("full_2024", frames["full_2024"], 2024),
        )
    }
    masks = {axis: rcore(frame) for axis, frame in frames.items()}
    frame22 = frames["full_2022"]
    early22 = frame22["game_month"].le(7).to_numpy() & masks["full_2022"]
    late22 = frame22["game_month"].ge(8).to_numpy() & masks["full_2022"]

    grid: list[dict[str, Any]] = []
    payload: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        columns = family_columns(features["full_2022"], family)
        model_early = fit_model(
            features["full_2022"].loc[early22, columns],
            frame22.loc[early22, TARGET].to_numpy(np.float64) - parents["full_2022"][early22],
        )
        direction_late22 = correction(
            model_early, features["full_2022"].loc[late22, columns]
        )
        model22 = fit_model(
            features["full_2022"].loc[masks["full_2022"], columns],
            frame22.loc[masks["full_2022"], TARGET].to_numpy(np.float64)
            - parents["full_2022"][masks["full_2022"]],
        )
        direction23 = correction(
            model22, features["late_2023"].loc[masks["late_2023"], columns]
        )
        residual_late22 = (
            frame22.loc[late22, TARGET].to_numpy(np.float64) - parents["full_2022"][late22]
        )
        residual23 = (
            frames["late_2023"].loc[masks["late_2023"], TARGET].to_numpy(np.float64)
            - parents["late_2023"][masks["late_2023"]]
        )
        dose = analytic_dose(
            [residual_late22, residual23], [direction_late22, direction23]
        )
        candidate_late22 = np.clip(
            parents["full_2022"][late22] + dose * direction_late22, 0.001, 0.999
        )
        candidate23 = apply(
            parents["late_2023"], masks["late_2023"], direction23, dose
        )
        source = {
            "early_to_late_2022": axis_metrics(
                frame22.loc[late22].reset_index(drop=True),
                parents["full_2022"][late22], candidate_late22,
                np.ones(int(late22.sum()), dtype=bool),
            ),
            "full_2022_to_late_2023": axis_metrics(
                frames["late_2023"], parents["late_2023"], candidate23,
                masks["late_2023"],
            ),
        }
        passed = bool(
            dose > 0.0
            and all(item["gain"] > 0.0 for item in source.values())
            and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source.values())
            and all(item["worst_month_gain"] > -10.0 for item in source.values())
        )
        row = {
            "family": family,
            "feature_count": len(columns),
            "dose": dose,
            "minimum_source_gain": min(item["gain"] for item in source.values()),
            "mean_source_gain": float(np.mean([item["gain"] for item in source.values()])),
            "source_pass": passed,
            "source_metrics": source,
        }
        grid.append(row)
        payload[family] = {
            "columns": columns, "candidate23": candidate23,
            "source": source, "dose": dose,
        }
    passing = [row for row in grid if row["source_pass"]]
    passing.sort(
        key=lambda row: (row["minimum_source_gain"], row["mean_source_gain"]),
        reverse=True,
    )
    restrictions = {
        "official_train_only": True,
        "predeclared_six_family_source_ablation": True,
        "family_and_dose_selected_on_two_sources_only": True,
        "full2024_opened_for_one_source_selected_family": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "source_grid": grid, "locked_origin_opened": False,
            "eligible_for_packaging": False, "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing[0]
    family = str(chosen["family"])
    columns = payload[family]["columns"]
    dose = float(chosen["dose"])
    combined_features = pd.concat(
        [
            features["full_2022"].loc[masks["full_2022"], columns],
            features["late_2023"].loc[masks["late_2023"], columns],
        ], ignore_index=True,
    )
    combined_residual = np.concatenate(
        [
            frame22.loc[masks["full_2022"], TARGET].to_numpy(np.float64)
            - parents["full_2022"][masks["full_2022"]],
            frames["late_2023"].loc[masks["late_2023"], TARGET].to_numpy(np.float64)
            - parents["late_2023"][masks["late_2023"]],
        ]
    )
    locked_model = fit_model(combined_features, combined_residual)
    direction24 = correction(
        locked_model, features["full_2024"].loc[masks["full_2024"], columns]
    )
    candidate24 = apply(
        parents["full_2024"], masks["full_2024"], direction24, dose
    )
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], candidate24, masks["full_2024"]
    )
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
        axes24, parents["full_2024"], candidate24, masks["full_2024"],
        [parents["full_2024"], candidate24],
    )
    passed = bool(
        locked["gain"] >= 1.5
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0004
        and robustness["chronological_block"]["p05"] > 0.0
    )
    joblib.dump(
        {"model": locked_model, "columns": columns, "dose": dose},
        output_dir / "locked_model.joblib", compress=3,
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[family]["candidate23"],
        parent_full_2024=parents["full_2024"], candidate_full_2024=candidate24,
        active_full_2024=masks["full_2024"], direction_full_2024=increment,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if passed else "locked_reject",
        "source_grid": grid, "selected_source_family": chosen,
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
