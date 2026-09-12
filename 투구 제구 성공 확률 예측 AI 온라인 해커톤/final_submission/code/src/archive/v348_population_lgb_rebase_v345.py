"""Rebase the frozen v293 ID-free F population expert above Public v345.

v293 selected ``consensus_add`` using early-to-late 2023 before v345 existed.
This audit carries that exact additive direction and dose onto the v345
analogue.  No model, route, active rows, or weight is reselected here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V348_POPULATION_LGB_REBASE_V345_V1"
TARGET = "control_success"


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        np.asarray(candidate, dtype=np.float64)
        - np.asarray(parent, dtype=np.float64)
    ))))


def apply_frozen_direction(
    parent: np.ndarray,
    direction: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    mask = np.asarray(active, dtype=bool)
    output[mask] = np.clip(
        output[mask] + np.asarray(direction, dtype=np.float64)[mask],
        0.001,
        0.999,
    )
    return output


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    v293_axes: Path,
    v293_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "game_type", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    frames = {
        "late_2023": frame.loc[
            frame["season"].eq(2023) & frame["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": frame.loc[frame["season"].eq(2024)].reset_index(drop=True),
    }
    audit = json.loads(v293_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V293_FUTURES_POPULATION_LGB_DIVERSITY_V1":
        raise ValueError("unexpected v293 protocol")
    if audit.get("selected", {}).get("rule") != "consensus_add":
        raise ValueError("v293 frozen rule is not consensus_add")

    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {"late_2023": saved["candidate_late_2023"].astype(np.float64)}
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    with np.load(v293_axes, allow_pickle=False) as saved:
        directions = {
            axis: (
                saved[f"candidate_{axis}"].astype(np.float64)
                - saved[f"parent_{axis}"].astype(np.float64)
            )
            for axis in frames
        }
        actives = {
            axis: saved[f"active_{axis}"].astype(bool)
            for axis in frames
        }

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for axis in frames:
        if not (len(frames[axis]) == len(parents[axis]) == len(directions[axis])):
            raise ValueError(f"axis alignment mismatch: {axis}")
        if np.any(actives[axis] & ~frames[axis]["game_type"].astype(str).eq("F").to_numpy()):
            raise ValueError(f"v293 active rows escape F: {axis}")
        candidate = apply_frozen_direction(parents[axis], directions[axis], actives[axis])
        candidates[axis] = candidate
        axes = {
            "target": frames[axis][TARGET].to_numpy(np.float64),
            "game_month": frames[axis]["game_month"].to_numpy(np.int16),
            "pitcher_id": frames[axis]["pitcher_id"].to_numpy(),
            "batter_id": frames[axis]["batter_id"].to_numpy(),
            "exact_mask": np.ones(len(frames[axis]), dtype=bool),
        }
        metrics[axis] = paired_metrics(axes, parents[axis], candidate, actives[axis])
        metrics[axis]["full_row_rms_shift"] = full_row_rms(parents[axis], candidate)

    increment = candidates["full_2024"] - parents["full_2024"]
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
        axes24,
        parents["full_2024"],
        candidates["full_2024"],
        actives["full_2024"],
        [parents["full_2024"], candidates["full_2024"]],
    )
    passed = bool(
        metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2024"]["gain"] >= 0.75
        and metrics["full_2024"]["positive_month_fraction"] >= 0.50
        and metrics["full_2024"]["full_row_rms_shift"] >= 0.001
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=candidates["late_2023"],
        active_late_2023=actives["late_2023"],
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=candidates["full_2024"],
        active_full_2024=actives["full_2024"],
        direction_full_2024=directions["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "parent": "v345 exact analogue",
        "frozen_component": "v293 ID-free LightGBM consensus_add at fixed 0.10 dose",
        "metrics": metrics,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "candidate_gate_passed": passed,
        "restrictions": {
            "official_train_only": True,
            "pitcher_and_batter_ids_excluded_from_component": True,
            "rule_active_rows_and_dose_frozen_before_v345_public": True,
            "full_2024_not_used_to_select_recipe": True,
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
    parser.add_argument("--v293-axes", type=Path, required=True)
    parser.add_argument("--v293-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v335_axes, args.v345_axes, args.v293_axes,
        args.v293_summary, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
