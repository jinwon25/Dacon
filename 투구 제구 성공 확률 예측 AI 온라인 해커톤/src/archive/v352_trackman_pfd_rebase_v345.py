"""Rebase the frozen v38 TrackMan-PFD student correction above v345.

The v38 recipe was selected on 2024 June--July and audited on August--October
before v345 existed.  This module does not tune its route or dose: it applies
the exact saved v38 candidate-minus-parent increment to the matching late-2024
rows and measures whether that independent direction survives above v345.
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


PROTOCOL = "V352_TRACKMAN_PFD_REBASE_V345_V1"
TARGET = "control_success"


def frozen_increment(v38_parent: np.ndarray, v38_candidate: np.ndarray) -> np.ndarray:
    """Return the exact, already-selected v38 probability increment."""
    return np.asarray(v38_candidate, dtype=np.float64) - np.asarray(
        v38_parent, dtype=np.float64
    )


def add_frozen_increment(parent: np.ndarray, increment: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + np.asarray(increment, dtype=np.float64),
        0.001,
        0.999,
    )


def run(
    train_csv: Path,
    v345_axes: Path,
    v38_audit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    late_mask = full_2024["game_month"].ge(8).to_numpy()
    frame = full_2024.loc[late_mask].reset_index(drop=True)

    with np.load(v345_axes, allow_pickle=False) as saved:
        v345_full = saved["candidate_full_2024"].astype(np.float64)
        v335_full = saved["parent_full_2024"].astype(np.float64)
    with np.load(v38_audit, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        game_month = saved["game_month"].astype(np.int16)
        old_parent = saved["v27"].astype(np.float64)
        old_candidate = saved["candidate"].astype(np.float64)
        active = saved["active"].astype(bool)

    if len(v345_full) != len(full_2024):
        raise ValueError("v345 full-2024 axis length mismatch")
    if not np.array_equal(frame[TARGET].to_numpy(np.float64), target):
        raise ValueError("v38 target order does not match late-2024")
    if not np.array_equal(frame["game_month"].to_numpy(np.int16), game_month):
        raise ValueError("v38 month order does not match late-2024")

    parent = v345_full[late_mask]
    increment = frozen_increment(old_parent, old_candidate)
    candidate = add_frozen_increment(parent, increment)
    metrics = axis_metrics(frame, parent, candidate, active)
    metrics["full_row_rms_shift"] = float(np.sqrt(np.mean(np.square(increment))))
    metrics["increment_correlation_vs_v345"] = float(
        np.corrcoef(increment, (v345_full - v335_full)[late_mask])[0, 1]
    )

    axes = {
        "target": target,
        "game_month": game_month,
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    robustness = _robustness(
        axes, parent, candidate, active, [parent, candidate]
    )
    passed = bool(
        metrics["gain"] >= 0.5
        and metrics["active_gain"] > 0.0
        and metrics["positive_month_fraction"] == 1.0
        and metrics["worst_month_gain"] >= 0.0
        and abs(metrics["increment_correlation_vs_v345"]) <= 0.10
    )

    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2024=parent,
        candidate_late_2024=candidate,
        increment_late_2024=increment,
        active_late_2024=active,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "reject",
        "recipe": {
            "parent": "submit_v345.zip / Public 1182.94969702",
            "increment": "frozen v38 without-ID TrackMan-PFD student ensemble",
            "domain": "R_ANCHOR",
            "weight": 0.40,
            "route_or_weight_retuned": False,
            "student_fit_window": "2024 March-July",
            "student_audit_window": "2024 August-October",
        },
        "metrics_vs_v345": metrics,
        "locked_robustness": robustness,
        "support": {
            "late_2024_rows": int(len(frame)),
            "active_rows": int(active.sum()),
        },
        "candidate_gate_passed": passed,
        "selection_warning": (
            "The v38 route and weight were frozen before this rebase; the "
            "late-2024 audit is independent of its June-July selection but "
            "was inspected historically in the v38 study."
        ),
        "restrictions": {
            "official_train_and_trackman_only": True,
            "current_pitch_trackman_at_inference": False,
            "student_uses_player_ids": False,
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
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--v38-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.v345_axes,
                args.v38_audit,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
