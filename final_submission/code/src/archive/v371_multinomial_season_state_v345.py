"""Strict-forward multinomial current-season command state above v345.

Official cumulative pitcher rates permit a row-local reconstruction of how
the current season differs from the pitcher's opening career state.  This
experiment models only those success/reverse/middle/ball/strike innovations
and their fixed count interactions.  It never inspects another evaluation
row.  Early-to-late 2022 and full-2022-to-late-2023 are source transfers;
full-2024 remains locked unless both source transfers are stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V371_MULTINOMIAL_SEASON_STATE_V345_V1"
TARGET = "control_success"
RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)
RIDGE_ALPHA = 1000.0
STATE_CONCENTRATION = 150.0
CORRECTION_CAP = 0.02
MAX_DOSE = 0.50


def opening_state(train: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return the final observable pre-pitch state from seasons before year."""

    history = train.loc[train["season"].lt(year)]
    if history.empty:
        return pd.DataFrame()
    latest = history.loc[
        history.groupby("pitcher_id", observed=True, sort=False)["asof_pitcher_n"].idxmax()
    ].copy()
    columns = ["pitcher_id", "asof_pitcher_n", *RATE_COLUMNS]
    latest = latest[columns].set_index("pitcher_id")
    return latest.rename(
        columns={
            "asof_pitcher_n": "opening_n",
            **{column: f"opening_{column}" for column in RATE_COLUMNS},
        }
    )


def state_features(train: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    opening = opening_state(train, year)
    identifiers = pd.to_numeric(rows["pitcher_id"], errors="raise")
    opening_n = identifiers.map(opening.get("opening_n", pd.Series(dtype=float)))
    career_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0)
    known = opening_n.notna().to_numpy()
    opening_n = opening_n.fillna(0.0).to_numpy(np.float64)
    career_n_array = career_n.to_numpy(np.float64)
    season_n = np.maximum(career_n_array - opening_n, 0.0)
    reliability = season_n / (season_n + STATE_CONCENTRATION)
    output = pd.DataFrame(index=np.arange(len(rows)))
    output["season_log_n"] = np.log1p(season_n)
    output["season_reliability"] = reliability
    output["returning_pitcher"] = known.astype(np.float64)

    innovations: dict[str, np.ndarray] = {}
    posteriors: dict[str, np.ndarray] = {}
    for column in RATE_COLUMNS:
        current_rate = pd.to_numeric(rows[column], errors="coerce").to_numpy(np.float64)
        global_prior = float(
            pd.to_numeric(
                train.loc[train["season"].eq(year - 1), column], errors="coerce"
            ).mean()
        )
        if not np.isfinite(global_prior):
            global_prior = 0.5 if column.endswith("success_rate") else 0.0
        opening_series = opening.get(f"opening_{column}", pd.Series(dtype=float))
        opening_rate = identifiers.map(opening_series).fillna(global_prior).to_numpy(np.float64)
        current_rate = np.where(np.isfinite(current_rate), current_rate, opening_rate)
        current_count = career_n_array * current_rate
        opening_count = opening_n * opening_rate
        season_count = np.clip(current_count - opening_count, 0.0, season_n)
        posterior = (
            season_count + STATE_CONCENTRATION * opening_rate
        ) / (season_n + STATE_CONCENTRATION)
        short = column.removeprefix("asof_pitcher_").removesuffix("_rate")
        innovation = posterior - opening_rate
        posteriors[short] = posterior
        innovations[short] = innovation
        output[f"state_{short}_posterior"] = posterior
        output[f"state_{short}_innovation"] = innovation
        output[f"state_{short}_vs_career"] = posterior - current_rate

    output["state_bad_location"] = (
        posteriors["reverse"] + posteriors["middle"]
    )
    output["state_called_result"] = posteriors["ball"] + posteriors["strike"]
    output["state_bad_location_innovation"] = (
        innovations["reverse"] + innovations["middle"]
    )
    output["state_ball_minus_strike_innovation"] = (
        innovations["ball"] - innovations["strike"]
    )
    balls = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(0).to_numpy()
    strikes = pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(0).to_numpy()
    three_ball = (balls >= 3).astype(np.float64)
    two_strike = (strikes >= 2).astype(np.float64)
    behind = (balls > strikes).astype(np.float64)
    for name in (
        "success", "reverse", "middle", "ball", "strike",
    ):
        value = innovations[name]
        output[f"state_{name}_innovation_x_three_ball"] = value * three_ball
        output[f"state_{name}_innovation_x_two_strike"] = value * two_strike
        output[f"state_{name}_innovation_x_behind"] = value * behind
    return output.astype(np.float64)


def rcore(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return regular & ~anchor


def fit_model(features: pd.DataFrame, residual: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)),
        ]
    )
    centered = np.asarray(residual, dtype=np.float64)
    centered -= float(centered.mean())
    model.fit(features, centered)
    return model


def correction(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict(features).astype(np.float64)
    return np.clip(raw - float(raw.mean()), -CORRECTION_CAP, CORRECTION_CAP)


def analytic_dose(residuals: list[np.ndarray], directions: list[np.ndarray]) -> float:
    numerator = sum(float(np.dot(r, d)) for r, d in zip(residuals, directions))
    denominator = sum(float(np.dot(d, d)) for d in directions)
    return float(np.clip(numerator / denominator, 0.0, MAX_DOSE)) if denominator else 0.0


def apply(parent: np.ndarray, active: np.ndarray, direction: np.ndarray, dose: float) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(output[active] + dose * direction, 0.001, 0.999)
    return output


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
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
    model_early = fit_model(
        features["full_2022"].loc[early22],
        frame22.loc[early22, TARGET].to_numpy(np.float64) - parents["full_2022"][early22],
    )
    direction_late22 = correction(model_early, features["full_2022"].loc[late22])
    model22 = fit_model(
        features["full_2022"].loc[masks["full_2022"]],
        frame22.loc[masks["full_2022"], TARGET].to_numpy(np.float64)
        - parents["full_2022"][masks["full_2022"]],
    )
    direction23 = correction(model22, features["late_2023"].loc[masks["late_2023"]])
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
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source.values())
        and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source.values())
        and all(item["worst_month_gain"] > -10.0 for item in source.values())
    )
    restrictions = {
        "official_train_only": True,
        "current_season_state_from_current_row_and_frozen_prior_snapshot": True,
        "identity_used_only_for_prior_train_lookup": True,
        "two_source_dose_selection": True,
        "full2024_opened_only_after_source_gate": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not source_pass:
        summary = {
            "protocol": PROTOCOL, "status": "source_reject",
            "source_selected_dose": dose, "source_metrics": source,
            "locked_origin_opened": False, "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    combined_features = pd.concat(
        [
            features["full_2022"].loc[masks["full_2022"]],
            features["late_2023"].loc[masks["late_2023"]],
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
        locked_model, features["full_2024"].loc[masks["full_2024"]]
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
    joblib.dump(locked_model, output_dir / "locked_model.joblib", compress=3)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parents["late_2023"], candidate_late_2023=candidate23,
        parent_full_2024=parents["full_2024"], candidate_full_2024=candidate24,
        active_full_2024=masks["full_2024"], direction_full_2024=increment,
    )
    summary = {
        "protocol": PROTOCOL, "status": "candidate" if passed else "locked_reject",
        "source_selected_dose": dose, "source_metrics": source,
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
