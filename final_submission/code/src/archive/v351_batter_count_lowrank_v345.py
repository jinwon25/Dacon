"""Strict-forward batter-by-count/pitcher-hand low-rank residual above v345.

This mirrors the successful *principle* of the pitcher low-rank component but
uses a disjoint entity and opposite-hand context.  Every matrix is fit from a
completed season's forward OOF residual, centred within that source season.
The exact smoothing/rank/route/dose recipe is selected jointly on full-2022
and late-2023 before full-2024 is opened once above the exact v345 analogue.
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


PROTOCOL = "V351_BATTER_COUNT_LOWRANK_V345_V1"
TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
SMOOTHING_GRID = (300.0, 600.0)
RANK_GRID = (2, 4)
ROUTES = ("ALL", "F", "R_ANCHOR", "R_CORE")
WEIGHTS = (0.25, 0.50)


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    balls = pd.to_numeric(output["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(output["strikes_before"], errors="raise").to_numpy(np.int16)
    hand = pd.to_numeric(output["pitcher_hand"], errors="raise").to_numpy(np.int16)
    if not (
        np.isin(balls, np.arange(4)).all()
        and np.isin(strikes, np.arange(3)).all()
        and np.isin(hand, (1, 2)).all()
    ):
        raise ValueError("unexpected count or pitcher-hand value")
    output["context_position"] = ((balls * 3 + strikes) * 2 + (hand - 1)).astype(np.int8)
    return output


def fit_source_matrix(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
) -> dict[str, Any]:
    source = prepare(rows).reset_index(drop=True)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    residual -= float(residual.mean())
    codes, batter_ids = pd.factorize(source["batter_id"], sort=True)
    contexts = source["context_position"].to_numpy(np.int16)
    shape = (len(batter_ids), 24)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(sums, (codes, contexts), residual)
    np.add.at(counts, (codes, contexts), 1)
    reconstructions: dict[tuple[float, int], np.ndarray] = {}
    for smoothing in SMOOTHING_GRID:
        saturated = sums / (counts.astype(np.float64) + float(smoothing))
        left, singular, right = np.linalg.svd(saturated, full_matrices=False)
        for rank in RANK_GRID:
            effective = min(int(rank), len(singular))
            reconstructions[(float(smoothing), int(rank))] = (
                left[:, :effective] * singular[:effective]
            ) @ right[:effective, :]
    return {
        "batter_ids": np.asarray(batter_ids),
        "counts": counts,
        "reconstructions": reconstructions,
        "rows": int(len(source)),
        "batters": int(len(batter_ids)),
    }


def map_source_matrix(model: dict[str, Any], rows: pd.DataFrame) -> dict[str, np.ndarray]:
    query = prepare(rows).reset_index(drop=True)
    batter_ids = np.asarray(model["batter_ids"])
    index = pd.Index(batter_ids).get_indexer(query["batter_id"])
    seen = index >= 0
    contexts = query["context_position"].to_numpy(np.int16)
    output: dict[str, np.ndarray] = {}
    for (smoothing, rank), matrix in model["reconstructions"].items():
        values = np.zeros(len(query), dtype=np.float64)
        values[seen] = np.asarray(matrix)[index[seen], contexts[seen]]
        output[f"batter_lowrank_s{int(smoothing)}_r{int(rank)}"] = values
    return output


def build_bank(
    models: dict[int, dict[str, Any]],
    rows: pd.DataFrame,
    audit_year: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    years = [year for year in SOURCE_YEARS if year < int(audit_year)]
    mapped = [map_source_matrix(models[year], rows) for year in years]
    names = sorted(mapped[0])
    return {
        name: np.mean(np.vstack([item[name] for item in mapped]), axis=0)
        for name in names
    }, years


def route_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return {
        "ALL": np.ones(len(frame), dtype=bool),
        "F": frame["game_type"].astype(str).eq("F").to_numpy(),
        "R_ANCHOR": anchor,
        "R_CORE": regular & ~anchor,
    }


def candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    signal: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = route_masks(frame)[route] & (np.abs(signal) > 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * signal[active], 0.001, 0.999
    )
    return output, active


def full_row_rms(parent: np.ndarray, candidate_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate_values - parent))))


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
        "pitcher_team_id", "batter_team_id", "pitcher_hand",
        "balls_before", "strikes_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    season_rows = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (*SOURCE_YEARS, 2024)
    }
    models: dict[int, dict[str, Any]] = {}
    model_diagnostics: dict[str, Any] = {}
    for year in SOURCE_YEARS:
        with np.load(oof_dir / f"wave0_incumbent_validate_{year}.npz", allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        expected = season_rows[year][TARGET].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        models[year] = fit_source_matrix(season_rows[year], target, parent)
        model_diagnostics[str(year)] = {
            "rows": models[year]["rows"],
            "batters": models[year]["batters"],
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
        full_bank, bank_years[axis] = build_bank(models, frames[axis], year)
        banks[axis] = full_bank

    rows: list[dict[str, Any]] = []
    payload: dict[tuple[str, str, float], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for signal_name in sorted(banks["full_2022"]):
        for route in ROUTES:
            for weight in WEIGHTS:
                key = (signal_name, route, float(weight))
                payload[key] = {}
                record: dict[str, Any] = {
                    "signal": signal_name, "route": route, "weight": float(weight)
                }
                for axis in ("full_2022", "late_2023"):
                    values, active = candidate(
                        frames[axis], parents[axis], banks[axis][signal_name], route, weight
                    )
                    payload[key][axis] = (values, active)
                    result = axis_metrics(frames[axis], parents[axis], values, active)
                    record[f"{axis}_gain"] = result["gain"]
                    record[f"{axis}_positive_month_fraction"] = result["positive_month_fraction"]
                    record[f"{axis}_worst_month_gain"] = result["worst_month_gain"]
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
                rows.append(record)
    grid = pd.DataFrame(rows).sort_values(
        ["source_pass", "minimum_source_gain", "full_2022_gain", "late_2023_gain"],
        ascending=False,
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "source_grid.csv", index=False)
    passing = grid.loc[grid["source_pass"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_grid_size": int(len(grid)),
            "source_passing_recipes": 0,
            "best_source_recipe": grid.iloc[0].to_dict(),
            "model_diagnostics": model_diagnostics,
            "candidate_gate_passed": False,
            "restrictions": {
                "official_train_only": True,
                "source_residuals_forward_oof": True,
                "strictly_prior_completed_seasons": True,
                "recipe_selected_before_full_2024": True,
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
    chosen = passing.iloc[0]
    recipe = {
        "signal": str(chosen["signal"]),
        "route": str(chosen["route"]),
        "weight": float(chosen["weight"]),
    }
    locked_candidate, locked_active = candidate(
        frames["full_2024"], parents["full_2024"],
        banks["full_2024"][recipe["signal"]], recipe["route"], recipe["weight"],
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
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        direction_full_2024=banks["full_2024"][recipe["signal"]],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "selected_recipe": recipe,
        "source_evidence": {
            "full_2022_gain": float(chosen["full_2022_gain"]),
            "late_2023_gain": float(chosen["late_2023_gain"]),
            "full_2022_positive_month_fraction": float(chosen["full_2022_positive_month_fraction"]),
            "late_2023_positive_month_fraction": float(chosen["late_2023_positive_month_fraction"]),
        },
        "source_grid_size": int(len(grid)),
        "source_passing_recipes": int(len(passing)),
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "bank_source_years": bank_years,
        "model_diagnostics": model_diagnostics,
        "candidate_gate_passed": passed,
        "restrictions": {
            "official_train_only": True,
            "source_residuals_forward_oof": True,
            "strictly_prior_completed_seasons": True,
            "recipe_selected_before_full_2024": True,
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
