"""Audit a frozen Beta-Binomial cell on top of the v343 portfolio.

The Beta cell, blend weight, and baseball archetype were selected before this
combination was evaluated.  This module only adds the already-frozen v337
increment to the already-frozen v343 predictions and records the paired
full-2024 diagnostics.  Full-2024 is development-contaminated evidence.
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


PROTOCOL = "V345_TRANSITION_WORKLOAD_BETA_CELL_V335_V1"
TARGET = "control_success"


def add_frozen_beta_increment(
    v343: np.ndarray,
    beta_parent: np.ndarray,
    beta_candidate: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(beta_candidate, dtype=np.float64) - np.asarray(
        beta_parent, dtype=np.float64
    )
    return np.clip(np.asarray(v343, dtype=np.float64) + delta, 0.001, 0.999)


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
    v343_axes: Path,
    beta_axes: Path,
    beta_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    frame = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    with np.load(v343_axes, allow_pickle=False) as saved:
        parent = saved["parent_full_2024"].astype(np.float64)
        v343 = saved["candidate_full_2024"].astype(np.float64)
        v343_active = saved["active_full_2024"].astype(bool)
    with np.load(beta_axes, allow_pickle=False) as saved:
        beta_parent = saved["parent_full_2024"].astype(np.float64)
        beta_candidate = saved["candidate_full_2024"].astype(np.float64)
        beta_active = saved["active_full_2024"].astype(bool)
    if not (
        len(frame)
        == len(parent)
        == len(v343)
        == len(beta_parent)
        == len(beta_candidate)
    ):
        raise ValueError("full-2024 axis alignment mismatch")
    if not np.allclose(parent, beta_parent, rtol=0.0, atol=0.0):
        raise ValueError("v343 and Beta axes do not share the exact v335 parent")

    candidate = add_frozen_beta_increment(v343, beta_parent, beta_candidate)
    active = v343_active | beta_active
    metrics = axis_metrics(frame, parent, candidate, active)
    metrics["full_row_rms_shift"] = full_row_rms(parent, candidate)
    beta_increment = beta_candidate - beta_parent
    v343_increment = v343 - parent
    nonzero = (np.abs(beta_increment) > 1e-15) | (
        np.abs(v343_increment) > 1e-15
    )
    metrics["increment_correlation"] = float(
        np.corrcoef(v343_increment[nonzero], beta_increment[nonzero])[0, 1]
    )
    target = frame[TARGET].to_numpy(np.float64)
    rate = float(target.mean())
    metrics["gain_increment_vs_v343"] = float(
        100000.0
        * (
            np.mean(np.square(v343 - target))
            - np.mean(np.square(candidate - target))
        )
        / (rate * (1.0 - rate))
    )
    axes = {
        "target": target,
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    robustness = _robustness(
        axes,
        parent,
        candidate,
        active,
        [parent, v343, beta_candidate, candidate],
    )
    beta_evidence = json.loads(beta_summary.read_text(encoding="utf-8"))
    source_gains = {
        origin: float(beta_evidence["metrics"][origin]["gain"])
        for origin in ("full_2022", "late_2023")
    }
    passed = bool(
        metrics["gain"] >= 3.2
        and metrics["gain_increment_vs_v343"] > 0.0
        and metrics["positive_month_fraction"] >= 0.75
        and all(value > 0.0 for value in source_gains.values())
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent,
        candidate_full_2024=candidate,
        active_full_2024=active,
        v343_increment_full_2024=v343_increment,
        beta_increment_full_2024=beta_increment,
        beta_active_full_2024=beta_active,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "reject",
        "recipe": {
            "parent": "v343 transition + three-seed workload H1",
            "increment": "frozen v337 Beta-Binomial cell direction",
            "cell": "R_CORE|DEVELOPING|MIXED",
            "beta_weight": 0.10,
            "route_or_weight_retuned": False,
        },
        "metrics_vs_v335": metrics,
        "beta_source_gains_vs_v335": source_gains,
        "support": {
            "v343_rows": int(v343_active.sum()),
            "beta_rows": int(beta_active.sum()),
            "overlap_rows": int(np.count_nonzero(v343_active & beta_active)),
        },
        "locked_robustness": robustness,
        "candidate_gate_passed": passed,
        "selection_warning": (
            "full-2024 was exposed in both parent studies and is diagnostic, "
            "not an independent holdout"
        ),
        "restrictions": {
            "official_train_only": True,
            "frozen_v343": True,
            "frozen_beta_cell_and_weight": True,
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
    parser.add_argument("--v343-axes", type=Path, required=True)
    parser.add_argument("--beta-axes", type=Path, required=True)
    parser.add_argument("--beta-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.v343_axes,
                args.beta_axes,
                args.beta_summary,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
