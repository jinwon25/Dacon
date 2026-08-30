"""Isolate the v330 R_CORE player-transition increment above v335.

v330's final route applies the same frozen v50 low-rank direction as v335 on
R_ANCHOR and the player-transition direction on R_CORE.  F is protected.  The
only incremental component above v335 is therefore the R_CORE assignment.
The v330 router used all three development origins and failed its leave-one-
origin gate; this audit is inventory recovery, not a clean holdout claim.
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


PROTOCOL = "V338_V330_REBASE_V335_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
EXPERT = "player_transition"


def rebase_candidate(
    v335: np.ndarray,
    v330_parent: np.ndarray,
    v330_candidate: np.ndarray,
    assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    active = np.asarray(assignment).astype(str) == EXPERT
    candidate = np.asarray(v335, dtype=np.float64).copy()
    increment = np.asarray(v330_candidate, dtype=np.float64) - np.asarray(
        v330_parent, dtype=np.float64
    )
    candidate[active] = np.clip(
        candidate[active] + increment[active], 0.001, 0.999
    )
    outside_parity = float(
        np.max(
            np.abs(
                np.asarray(v330_candidate, dtype=np.float64)[~active]
                - np.asarray(v335, dtype=np.float64)[~active]
            )
        )
    )
    return candidate, active, outside_parity


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
    v330_axes: Path,
    v330_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "pitcher_id", "batter_id", TARGET,
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
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
    with np.load(v330_axes, allow_pickle=False) as saved:
        v330_parents = {
            origin: saved[f"parent_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        v330_candidates = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        assignments = {
            origin: saved[f"assignment_{origin}"].astype(str)
            for origin in ORIGINS
        }

    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        candidates[origin], active[origin], parity[origin] = rebase_candidate(
            parents[origin],
            v330_parents[origin],
            v330_candidates[origin],
            assignments[origin],
        )
        if parity[origin] > 1e-12:
            raise ValueError(f"v330/v335 protected-route parity failed: {origin}")
        metrics[origin] = axis_metrics(
            frames[origin], parents[origin], candidates[origin], active[origin]
        )
        metrics[origin]["full_row_rms_shift"] = full_row_rms(
            parents[origin], candidates[origin]
        )

    frame24 = frames["full_2024"]
    axes24 = {
        "target": frame24[TARGET].to_numpy(np.float64),
        "game_month": frame24["game_month"].to_numpy(np.int16),
        "pitcher_id": frame24["pitcher_id"].to_numpy(),
        "batter_id": frame24["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame24), dtype=bool),
    }
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        candidates["full_2024"],
        active["full_2024"],
        [parents["full_2024"], candidates["full_2024"]],
    )
    historical = json.loads(v330_summary.read_text(encoding="utf-8"))
    locked = metrics["full_2024"]
    candidate_gate = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.75
        and all(metrics[name]["gain"] > 0.0 for name in ORIGINS)
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in ORIGINS},
        **{f"candidate_{name}": candidates[name] for name in ORIGINS},
        **{f"active_{name}": active[name] for name in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "inventory_candidate" if candidate_gate else "inventory_reject",
        "recipe": {
            "parent": "v335",
            "increment": "v330 final router R_CORE player_transition only",
            "F_and_R_ANCHOR_preserved_exactly": True,
        },
        "metrics": metrics,
        "protected_route_parity_max_abs": parity,
        "locked_robustness": robustness,
        "rms_goal_context": {
            "locked_full_row_rms": locked["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": float(
                locked["full_row_rms_shift"] / 0.004551
            ),
        },
        "candidate_gate_passed": candidate_gate,
        "historical_selection_risk": {
            "v330_status": historical["status"],
            "v330_minimum_loo_gain": historical["selected_schema"][
                "minimum_loo_gain"
            ],
            "router_used_all_three_development_origins": True,
            "full_2024_is_not_a_fresh_holdout": True,
        },
        "restrictions": {
            "official_train_only": True,
            "v330_route_and_expert_frozen": True,
            "v335_F_and_R_ANCHOR_preserved": True,
            "crossed_bootstrap_and_reality_check_are_diagnostic_only": True,
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
    parser.add_argument("--v330-axes", type=Path, required=True)
    parser.add_argument("--v330-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v330_axes,
        args.v330_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
