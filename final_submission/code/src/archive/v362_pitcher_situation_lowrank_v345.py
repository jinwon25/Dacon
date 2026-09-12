"""Strict-forward pitcher-by-situation low-rank residual audit above v345.

The Public-improving v335 component factorises pitcher by count and batter
hand.  This audit applies the same frozen smoothing/rank/dose principle to a
small, baseball-defined library of *different* situational axes.  Matrices are
fit only from completed-season forward OOF residuals and averaged across
strictly prior seasons.  The source axes (full 2022 and late 2023) select one
schema before full 2024 is opened once.  Only R_CORE is modified so the v345 F
and team-13 paths remain exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import axis_metrics


PROTOCOL = "V362_PITCHER_SITUATION_LOWRANK_V345_V1"
TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
SMOOTHING = 300.0
RANK = 2
WEIGHT = 0.50
SCHEMA_NAMES = (
    "count_runner",
    "inning_hand",
    "outs_runner_count",
    "count_homeaway",
    "inning_pressure",
)


def situation_positions(rows: pd.DataFrame) -> dict[str, tuple[np.ndarray, int]]:
    balls = pd.to_numeric(rows["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(rows["strikes_before"], errors="raise").to_numpy(np.int16)
    if not (np.isin(balls, np.arange(4)).all() and np.isin(strikes, np.arange(3)).all()):
        raise ValueError("unexpected count value")
    count = balls * 3 + strikes

    runners = pd.to_numeric(rows["num_runners_on"], errors="coerce").fillna(0).to_numpy(np.int16)
    runners = np.clip(runners, 0, 3)
    runner_any = (runners > 0).astype(np.int16)
    outs = pd.to_numeric(rows["outs_before"], errors="raise").to_numpy(np.int16)
    if not np.isin(outs, np.arange(3)).all():
        raise ValueError("unexpected outs value")

    inning = pd.to_numeric(rows["inning"], errors="coerce").fillna(1).to_numpy(np.float64)
    inning_phase = np.select((inning <= 3, inning <= 6), (0, 1), default=2).astype(np.int16)
    batter_hand = pd.to_numeric(rows["batter_hand"], errors="raise").to_numpy(np.int16)
    if not np.isin(batter_hand, (1, 2)).all():
        raise ValueError("unexpected batter hand")
    hand = batter_hand - 1

    top_bottom = rows["top_bottom"].astype(str).str.upper()
    home_away = top_bottom.eq("B").astype(np.int16).to_numpy()
    leverage = pd.to_numeric(rows["li"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    pressure = (leverage >= 1.5).astype(np.int16)

    return {
        "count_runner": ((count * 2 + runner_any).astype(np.int16), 24),
        "inning_hand": ((inning_phase * 2 + hand).astype(np.int16), 6),
        "outs_runner_count": ((outs * 4 + runners).astype(np.int16), 12),
        "count_homeaway": ((count * 2 + home_away).astype(np.int16), 24),
        "inning_pressure": ((inning_phase * 2 + pressure).astype(np.int16), 6),
    }


def fit_source_matrix(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    schema: str,
) -> dict[str, Any]:
    positions, width = situation_positions(rows)[schema]
    residual = np.asarray(target, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    residual -= float(residual.mean())
    codes, pitcher_ids = pd.factorize(rows["pitcher_id"], sort=True)
    sums = np.zeros((len(pitcher_ids), width), dtype=np.float64)
    counts = np.zeros((len(pitcher_ids), width), dtype=np.int64)
    np.add.at(sums, (codes, positions), residual)
    np.add.at(counts, (codes, positions), 1)
    saturated = sums / (counts.astype(np.float64) + SMOOTHING)
    left, singular, right = np.linalg.svd(saturated, full_matrices=False)
    effective_rank = min(RANK, len(singular))
    reconstruction = (
        left[:, :effective_rank] * singular[:effective_rank]
    ) @ right[:effective_rank, :]
    total_energy = float(np.square(singular).sum())
    retained_energy = float(np.square(singular[:effective_rank]).sum())
    return {
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "reconstruction": reconstruction,
        "width": int(width),
        "rows": int(len(rows)),
        "pitchers": int(len(pitcher_ids)),
        "observed_cells": int(np.count_nonzero(counts)),
        "retained_energy": retained_energy / total_energy if total_energy > 0.0 else 0.0,
    }


def map_source_matrix(
    model: dict[str, Any], rows: pd.DataFrame, schema: str
) -> tuple[np.ndarray, np.ndarray]:
    positions, width = situation_positions(rows)[schema]
    if width != int(model["width"]):
        raise ValueError("context width mismatch")
    index = pd.Index(np.asarray(model["pitcher_ids"])).get_indexer(rows["pitcher_id"])
    seen = index >= 0
    values = np.zeros(len(rows), dtype=np.float64)
    values[seen] = np.asarray(model["reconstruction"])[index[seen], positions[seen]]
    return values, seen


def build_bank(
    models: dict[str, dict[int, dict[str, Any]]],
    rows: pd.DataFrame,
    audit_year: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    years = [year for year in SOURCE_YEARS if year < int(audit_year)]
    if not years:
        raise ValueError("no prior completed source season")
    bank: dict[str, np.ndarray] = {}
    for schema in SCHEMA_NAMES:
        mapped = [map_source_matrix(models[schema][year], rows, schema)[0] for year in years]
        bank[schema] = np.mean(np.vstack(mapped), axis=0)
    return bank, years


def rcore_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return regular & ~anchor


def make_candidate(
    frame: pd.DataFrame, parent: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = rcore_mask(frame) & np.not_equal(signal, 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + WEIGHT * np.asarray(signal, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output, active


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(candidate) - np.asarray(parent)))))


def run(
    train_csv: Path,
    oof_dir: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "batter_hand", "balls_before",
        "strikes_before", "inning", "top_bottom", "outs_before",
        "num_runners_on", "li", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    season_rows = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (*SOURCE_YEARS, 2024)
    }

    models: dict[str, dict[int, dict[str, Any]]] = {schema: {} for schema in SCHEMA_NAMES}
    model_diagnostics: dict[str, Any] = {schema: {} for schema in SCHEMA_NAMES}
    for year in SOURCE_YEARS:
        with np.load(oof_dir / f"wave0_incumbent_validate_{year}.npz", allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            incumbent = saved["incumbent"].astype(np.float64)
        expected = season_rows[year][TARGET].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        for schema in SCHEMA_NAMES:
            fitted = fit_source_matrix(season_rows[year], target, incumbent, schema)
            models[schema][year] = fitted
            model_diagnostics[schema][str(year)] = {
                key: fitted[key]
                for key in ("rows", "pitchers", "width", "observed_cells", "retained_energy")
            }

    frames = {
        "full_2022": season_rows[2022],
        "late_2023": season_rows[2023].loc[
            season_rows[2023]["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": season_rows[2024],
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

    banks: dict[str, dict[str, np.ndarray]] = {}
    bank_years: dict[str, list[int]] = {}
    for axis, year in (("full_2022", 2022), ("late_2023", 2023), ("full_2024", 2024)):
        banks[axis], bank_years[axis] = build_bank(models, frames[axis], year)

    source_rows: list[dict[str, Any]] = []
    source_payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for schema in SCHEMA_NAMES:
        source_payload[schema] = {}
        record: dict[str, Any] = {"schema": schema}
        for axis in ("full_2022", "late_2023"):
            candidate, active = make_candidate(frames[axis], parents[axis], banks[axis][schema])
            source_payload[schema][axis] = (candidate, active)
            result = axis_metrics(frames[axis], parents[axis], candidate, active)
            record[f"{axis}_gain"] = result["gain"]
            record[f"{axis}_positive_month_fraction"] = result["positive_month_fraction"]
            record[f"{axis}_worst_month_gain"] = result["worst_month_gain"]
            record[f"{axis}_rms"] = full_row_rms(parents[axis], candidate)
        record["minimum_source_gain"] = min(
            record["full_2022_gain"], record["late_2023_gain"]
        )
        record["source_pass"] = bool(
            record["full_2022_gain"] > 0.0
            and record["late_2023_gain"] > 0.0
            and record["full_2022_positive_month_fraction"] >= 4.0 / 7.0
            and record["late_2023_positive_month_fraction"] >= 2.0 / 3.0
            and record["full_2022_worst_month_gain"] > -10.0
            and record["late_2023_worst_month_gain"] > -10.0
        )
        source_rows.append(record)

    grid = pd.DataFrame(source_rows).sort_values(
        ["source_pass", "minimum_source_gain", "full_2022_gain", "late_2023_gain"],
        ascending=False,
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "source_grid.csv", index=False, encoding="utf-8-sig")
    passing = grid.loc[grid["source_pass"]]
    common_restrictions = {
        "official_train_only": True,
        "source_residuals_forward_oof": True,
        "strictly_prior_completed_seasons": True,
        "fixed_smoothing_rank_route_and_dose": True,
        "schema_selected_before_full_2024": True,
        "v345_f_and_team13_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "fixed_recipe": {
                "smoothing": SMOOTHING, "rank": RANK, "weight": WEIGHT, "route": "R_CORE"
            },
            "source_screen": source_rows,
            "locked_origin_opened": False,
            "model_diagnostics": model_diagnostics,
            "eligible_for_packaging": False,
            "restrictions": common_restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing.iloc[0]
    schema = str(chosen["schema"])
    locked_candidate, locked_active = make_candidate(
        frames["full_2024"], parents["full_2024"], banks["full_2024"][schema]
    )
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], locked_candidate, locked_active
    )
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], locked_candidate)
    increment = locked_candidate - parents["full_2024"]
    nonzero = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = (
        float(np.corrcoef(increment[nonzero], v345_increment[nonzero])[0, 1])
        if int(nonzero.sum()) > 2
        and float(np.std(increment[nonzero])) > 0.0
        and float(np.std(v345_increment[nonzero])) > 0.0
        else 0.0
    )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], locked_candidate, locked_active,
        [parents["full_2024"], locked_candidate],
    )
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{
            f"parent_{axis}": parents[axis]
            for axis in ("full_2022", "late_2023", "full_2024")
        },
        **{
            f"candidate_{axis}": (
                source_payload[schema][axis][0]
                if axis != "full_2024" else locked_candidate
            )
            for axis in ("full_2022", "late_2023", "full_2024")
        },
        active_full_2024=locked_active,
        direction_full_2024=banks["full_2024"][schema],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "fixed_recipe": {
            "smoothing": SMOOTHING, "rank": RANK, "weight": WEIGHT, "route": "R_CORE"
        },
        "selected_schema": schema,
        "source_screen": source_rows,
        "source_passing_schemas": int(len(passing)),
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "bank_source_years": bank_years,
        "model_diagnostics": model_diagnostics,
        "eligible_for_packaging": passed,
        "restrictions": common_restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.oof_dir, args.v335_axes, args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
