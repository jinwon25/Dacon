"""Pre-frozen F-route batter-transition residual above the Public v345 parent.

The recipe was one of only two batter-only recipes that passed both v324
source origins before v345 was submitted.  The primary dose is fixed to the
source-preferred 0.50; full-2024 is opened once only as a locked development
audit above the exact v345 OOF analogue.  The 0.25 dose is reported solely as
a sensitivity diagnostic and cannot replace the primary recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import (
    _candidate,
    _route_mask,
    axis_metrics,
    build_signal_bank,
)


PROTOCOL = "V347_BATTER_TRANSITION_REBASE_V345_V1"
TARGET = "control_success"
ALPHA = 250.0
SIGNAL = "batter_status_count"
ROUTE = "F"
PRIMARY_WEIGHT = 0.50
SENSITIVITY_WEIGHT = 0.25
SOURCE_EVIDENCE = {
    "primary": {"full_2022_gain": 1.0203845296819054, "late_2023_gain": 0.5018700850345681},
    "sensitivity": {"full_2022_gain": 0.7121020338395283, "late_2023_gain": 0.4397280794710241},
}


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        np.asarray(candidate, dtype=np.float64)
        - np.asarray(parent, dtype=np.float64)
    ))))


def run(
    train_csv: Path,
    oof_dir: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frame = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    with np.load(v345_axes, allow_pickle=False) as saved:
        parent = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    if len(frame) != len(parent):
        raise ValueError("v345 parent/full-2024 alignment mismatch")

    bank, bank_diagnostics = build_signal_bank(
        train, frame, 2024, oof_dir, ALPHA,
    )
    active = _route_mask(frame, ROUTE)
    signal = bank[SIGNAL]
    primary = _candidate(parent, signal, active, PRIMARY_WEIGHT)
    sensitivity = _candidate(parent, signal, active, SENSITIVITY_WEIGHT)
    primary_metrics = axis_metrics(frame, parent, primary, active)
    primary_metrics["full_row_rms_shift"] = full_row_rms(parent, primary)
    sensitivity_metrics = axis_metrics(frame, parent, sensitivity, active)
    sensitivity_metrics["full_row_rms_shift"] = full_row_rms(parent, sensitivity)

    increment = primary - parent
    nonzero = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = (
        float(np.corrcoef(increment[nonzero], v345_increment[nonzero])[0, 1])
        if int(nonzero.sum()) > 2
        and float(np.std(increment[nonzero])) > 0.0
        and float(np.std(v345_increment[nonzero])) > 0.0
        else 0.0
    )
    axes24 = {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    robustness = _robustness(
        axes24, parent, primary, active, [parent, primary],
    )
    passed = bool(
        primary_metrics["gain"] >= 0.50
        and primary_metrics["positive_month_fraction"] >= 0.625
        and primary_metrics["worst_month_gain"] > -5.0
        and primary_metrics["full_row_rms_shift"] >= 0.00030
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent,
        candidate_full_2024=primary,
        sensitivity_candidate_full_2024=sensitivity,
        active_full_2024=active,
        direction_full_2024=signal,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "recipe": {
            "alpha": ALPHA,
            "signal": SIGNAL,
            "route": ROUTE,
            "primary_weight": PRIMARY_WEIGHT,
            "sensitivity_weight": SENSITIVITY_WEIGHT,
        },
        "source_evidence_frozen_before_v345_public": SOURCE_EVIDENCE,
        "locked_primary_full_2024": primary_metrics,
        "locked_sensitivity_full_2024": sensitivity_metrics,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "bank_diagnostics": bank_diagnostics,
        "candidate_gate_passed": passed,
        "selection_warning": (
            "The primary weight is frozen from v324 source origins; the "
            "sensitivity result is diagnostic and is not eligible for selection."
        ),
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_season_entity_state": True,
            "source_residuals_forward_oof": True,
            "recipe_frozen_before_v345_public": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_recipe_or_dose": False,
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
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.oof_dir, args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
