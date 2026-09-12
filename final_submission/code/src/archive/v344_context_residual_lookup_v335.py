"""Audit a frozen ID-free baseball-context residual lookup above v335.

Corrections are estimated on full-2022 R_CORE residuals only.  Three fixed,
baseball-defined cell schemas are ranked on late-2023; full-2024 is evaluated
only if the source gate passes.  No pitcher or batter identity is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V344_CONTEXT_RESIDUAL_LOOKUP_V335_V1"
TARGET = "control_success"
ALPHA = 500.0
DOSE = 0.25
SCHEMAS = {
    "count_hand": ("balls_before", "strikes_before", "same_hand"),
    "count_pressure_hand": (
        "balls_before",
        "strikes_before",
        "runner_bin",
        "high_li",
        "same_hand",
    ),
    "count_base_hand": (
        "balls_before",
        "strikes_before",
        "base_state",
        "same_hand",
    ),
}


def context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["balls_before"] = pd.to_numeric(
        frame["balls_before"], errors="coerce"
    ).fillna(-1).astype(np.int16)
    output["strikes_before"] = pd.to_numeric(
        frame["strikes_before"], errors="coerce"
    ).fillna(-1).astype(np.int16)
    output["base_state"] = frame["base_state"].fillna("__NA__").astype(str)
    output["same_hand"] = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).astype(np.int8)
    output["runner_bin"] = pd.to_numeric(
        frame["num_runners_on"], errors="coerce"
    ).fillna(0).gt(0).astype(np.int8)
    output["high_li"] = pd.to_numeric(
        frame["li"], errors="coerce"
    ).fillna(0.0).ge(1.5).astype(np.int8)
    return output


def rcore_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return regular & ~anchor


def fit_lookup(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    columns: tuple[str, ...],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    features = context_frame(frame)
    active = rcore_mask(frame)
    fit = features.loc[active, list(columns)].copy()
    fit["residual"] = (
        np.asarray(target, dtype=np.float64)[active]
        - np.asarray(parent, dtype=np.float64)[active]
    )
    table = (
        fit.groupby(list(columns), sort=True, observed=True)["residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    table["correction"] = table["sum"] / (table["size"] + float(alpha))
    return table.drop(columns="sum")


def apply_lookup(
    frame: pd.DataFrame,
    parent: np.ndarray,
    table: pd.DataFrame,
    columns: tuple[str, ...],
    dose: float = DOSE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = context_frame(frame)
    mapped = features.loc[:, list(columns)].merge(
        table,
        on=list(columns),
        how="left",
        sort=False,
        validate="many_to_one",
    )
    correction = mapped["correction"].fillna(0.0).to_numpy(np.float64)
    active = rcore_mask(frame) & np.not_equal(correction, 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(dose) * correction[active], 0.001, 0.999
    )
    return output, active, correction


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(candidate, dtype=np.float64)
                    - np.asarray(parent, dtype=np.float64)
                )
            )
        )
    )


def run(
    train_csv: Path,
    v335_axes: Path,
    v339_axes: Path,
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
            for axis in frames
        }
    target22 = frames["full_2022"][TARGET].to_numpy(np.float64)
    source_rows: list[dict[str, Any]] = []
    source_candidates: dict[str, np.ndarray] = {}
    source_active: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    for schema, columns in SCHEMAS.items():
        table = fit_lookup(
            frames["full_2022"], target22, parents["full_2022"], columns
        )
        candidate, active, correction = apply_lookup(
            frames["late_2023"], parents["late_2023"], table, columns
        )
        metrics = axis_metrics(
            frames["late_2023"], parents["late_2023"], candidate, active
        )
        metrics["full_row_rms_shift"] = full_row_rms(
            parents["late_2023"], candidate
        )
        passed = bool(
            metrics["gain"] > 0.0
            and metrics["positive_month_fraction"] >= 2.0 / 3.0
            and metrics["worst_month_gain"] > -3.0
        )
        source_rows.append(
            {
                "schema": schema,
                "columns": list(columns),
                "cells": len(table),
                "mapped_rows": int(np.count_nonzero(correction)),
                "source_gate_passed": passed,
                **metrics,
            }
        )
        source_candidates[schema] = candidate
        source_active[schema] = active
        tables[schema] = table
    source_rows.sort(
        key=lambda row: (
            bool(row["source_gate_passed"]),
            float(row["gain"]),
            float(row["worst_month_gain"]),
        ),
        reverse=True,
    )
    pd.DataFrame(source_rows).to_csv(
        output_dir / "source_screen.csv", index=False, encoding="utf-8-sig"
    )
    passing = [row for row in source_rows if row["source_gate_passed"]]
    if not passing:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "alpha": ALPHA,
            "dose": DOSE,
            "source_screen": source_rows,
            "locked_origin_opened": False,
            "restrictions": {
                "official_train_only": True,
                "lookup_fit_full_2022_only": True,
                "schema_selected_late_2023_only": True,
                "full_2024_not_opened_after_source_failure": True,
                "identity_columns_excluded": True,
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

    selected = passing[0]
    schema = str(selected["schema"])
    columns = SCHEMAS[schema]
    component24, active24, correction24 = apply_lookup(
        frames["full_2024"], parents["full_2024"], tables[schema], columns
    )
    component_metrics = axis_metrics(
        frames["full_2024"], parents["full_2024"], component24, active24
    )
    component_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], component24
    )
    with np.load(v339_axes, allow_pickle=False) as saved:
        v339_parent = saved["parent_full_2024"].astype(np.float64)
        v339_candidate = saved["candidate_full_2024"].astype(np.float64)
        v339_active = saved["active_full_2024"].astype(bool)
    if np.max(np.abs(v339_parent - parents["full_2024"])) > 1e-12:
        raise ValueError("v339 parent is not v335")
    portfolio = v339_candidate.copy()
    portfolio[active24] = np.clip(
        portfolio[active24] + DOSE * correction24[active24], 0.001, 0.999
    )
    portfolio_active = v339_active | active24
    v339_metrics = axis_metrics(
        frames["full_2024"], parents["full_2024"], v339_candidate, v339_active
    )
    portfolio_metrics = axis_metrics(
        frames["full_2024"], parents["full_2024"], portfolio, portfolio_active
    )
    portfolio_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], portfolio
    )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        portfolio,
        portfolio_active,
        [parents["full_2024"], v339_candidate, component24, portfolio],
    )
    passed = bool(
        component_metrics["gain"] > 0.0
        and portfolio_metrics["gain"] > v339_metrics["gain"]
        and portfolio_metrics["positive_month_fraction"] >= 0.75
    )
    tables[schema].to_csv(
        output_dir / "selected_lookup.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=portfolio,
        active_full_2024=portfolio_active,
        context_increment_full_2024=component24 - parents["full_2024"],
        v339_increment_full_2024=v339_candidate - parents["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "inventory_candidate" if passed else "locked_reject",
        "alpha": ALPHA,
        "dose": DOSE,
        "selected_schema": schema,
        "selected_columns": list(columns),
        "source_screen": source_rows,
        "locked_origin_opened": True,
        "locked_component_metrics": component_metrics,
        "v339_metrics": v339_metrics,
        "portfolio_metrics": portfolio_metrics,
        "support": {
            "context_rows": int(active24.sum()),
            "v339_rows": int(v339_active.sum()),
            "overlap_rows": int(np.count_nonzero(active24 & v339_active)),
        },
        "locked_robustness": robustness,
        "candidate_gate_passed": passed,
        "selection_warning": (
            "full-2024 has been repeatedly exposed in earlier research; the "
            "source lookup and schema are frozen before this locked evaluation"
        ),
        "restrictions": {
            "official_train_only": True,
            "lookup_fit_full_2022_only": True,
            "schema_selected_late_2023_only": True,
            "full_2024_used_after_source_freeze": True,
            "identity_columns_excluded": True,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v339-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_csv, args.v335_axes, args.v339_axes, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
