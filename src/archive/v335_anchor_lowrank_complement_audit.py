"""Audit the frozen low-rank direction only on the stable R_ANCHOR route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V335_ANCHOR_LOWRANK_COMPLEMENT_AUDIT_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    v314_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
    with np.load(v314_axes, allow_pickle=False) as saved:
        lowrank_increment = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            - saved[f"parent_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }

    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        frame = frames[origin]
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        active[origin] = regular & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
        candidates[origin] = parents[origin].copy()
        candidates[origin][active[origin]] = np.clip(
            parents[origin][active[origin]]
            + lowrank_increment[origin][active[origin]],
            0.001,
            0.999,
        )
        metrics[origin] = axis_metrics(
            frame, parents[origin], candidates[origin], active[origin]
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
    sign_gate = all(metrics[origin]["gain"] > 0.0 for origin in ORIGINS)
    month_gate = all(
        metrics[origin]["positive_month_fraction"] >= 2.0 / 3.0
        for origin in ORIGINS
    )
    uncertainty_gate = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{f"candidate_{origin}": candidates[origin] for origin in ORIGINS},
        **{f"active_{origin}": active[origin] for origin in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if sign_gate and month_gate and uncertainty_gate else "reject",
        "recipe": {
            "base": "v320-equivalent axes",
            "route": "R_ANCHOR",
            "anchor_team": 13,
            "expert": "v50 lowrank_s300_r2 finalized through 2024",
            "dose": 0.50,
        },
        "metrics": metrics,
        "full_2024_robustness": robustness,
        "gates": {
            "all_origin_gain_positive": sign_gate,
            "all_origin_positive_month_fraction_ge_two_thirds": month_gate,
            "pitcher_chronological_and_reality_check": uncertainty_gate,
        },
        "selection_risk": (
            "route/expert was discovered in a multi-expert screen; the reported "
            "reality check covers only parent vs this frozen candidate"
        ),
        "restrictions": {
            "official_train_only": True,
            "frozen_historical_dose": True,
            "F_route_preserved_as_v320": True,
            "test_data_used": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--v314-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, args.v314_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
