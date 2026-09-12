"""One-shot source-only audit of an incremental v104-safe raw dose above v345."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v363_v104_inactive_raw_complement_v345 import (
    TARGET,
    axis_metrics,
    full_row_rms,
    load_raw_directions,
    rcore_mask,
    safe_votes,
)


PROTOCOL = "V364_V104_SAFE_INCREMENTAL_DOSE_V345_V1"
DOSES = (0.125, 0.25, 0.50)


def make_candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direction: np.ndarray,
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = rcore_mask(frame) & (safe_votes(frame) >= 2) & np.not_equal(direction, 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(dose) * np.asarray(direction)[active], 0.001, 0.999
    )
    return output, active


def run(
    train_csv: Path,
    v84_dir: Path,
    v103_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "asof_pitcher_n", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    directions = load_raw_directions(v84_dir, v103_axes)
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

    source_rows: list[dict[str, Any]] = []
    payload: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for dose in DOSES:
        payload[dose] = {}
        record: dict[str, Any] = {"dose": dose}
        for axis in ("full_2022", "late_2023"):
            candidate, active = make_candidate(
                frames[axis], parents[axis], directions[axis], dose
            )
            payload[dose][axis] = (candidate, active)
            metrics = axis_metrics(frames[axis], parents[axis], candidate, active)
            for key in ("gain", "positive_month_fraction", "worst_month_gain", "active_rows"):
                record[f"{axis}_{key}"] = metrics[key]
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
        ["source_pass", "minimum_source_gain", "dose"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "source_grid.csv", index=False, encoding="utf-8-sig")
    passing = grid.loc[grid["source_pass"]]
    restrictions = {
        "official_train_only": True,
        "v103_raw_direction_and_v104_safe_mask_frozen": True,
        "incremental_dose_selected_on_two_sources_only": True,
        "one_shot_family_audit": True,
        "full2024_opened_once_after_source_gate": True,
        "v345_f_anchor_and_v104_inactive_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_grid": source_rows,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing.iloc[0]
    dose = float(chosen["dose"])
    locked_candidate, locked_active = make_candidate(
        frames["full_2024"], parents["full_2024"], directions["full_2024"], dose
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
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[dose]["full_2022"][0],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[dose]["late_2023"][0],
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        raw_direction_full_2024=directions["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "selected_incremental_dose": dose,
        "source_grid": source_rows,
        "source_selected": chosen.to_dict(),
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v84-dir", type=Path, required=True)
    parser.add_argument("--v103-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v84_dir, args.v103_axes, args.v335_axes,
        args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
