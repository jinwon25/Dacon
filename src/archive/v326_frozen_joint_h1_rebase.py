"""Rebase the frozen v221 joint-H1 direction above the v320 portfolio.

The joint-H1 seed-42 screen improved full 2022, 2023 and 2024 above its own
paired H1 parent, with every 2024 month positive.  The later v221 experiment
was rejected because adding the role model reduced the locked gain.  This
one-shot audit does not revisit the rejected role blend or tune a new dose: it
maps the already frozen 10% joint-H1 direction onto R_CORE above v285/v320.
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


PROTOCOL = "V326_FROZEN_JOINT_H1_REBASE_V1"
TARGET = "control_success"
ANCHOR_TEAM = 13


def r_core(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(ANCHOR_TEAM)
        | frame["batter_team_id"].eq(ANCHOR_TEAM)
    ).to_numpy()
    return regular & ~anchor


def _axes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    v221_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_team_id", "batter_team_id",
        "pitcher_id", "batter_id", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    annual = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = annual[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": annual[2022],
        "late_2023": annual[2023].loc[late23].reset_index(drop=True),
        "full_2024": annual[2024],
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
        v320_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    with np.load(v221_axes, allow_pickle=False) as saved:
        annual_direction = {
            year: saved[f"candidate_{year}"].astype(np.float64)
            - saved[f"baseline_{year}"].astype(np.float64)
            for year in (2022, 2023, 2024)
        }
    directions = {
        "full_2022": annual_direction[2022],
        "late_2023": annual_direction[2023][late23],
        "full_2024": annual_direction[2024],
    }

    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        if not (len(frame) == len(parents[name]) == len(directions[name])):
            raise ValueError(f"axis alignment mismatch: {name}")
        active[name] = r_core(frame)
        candidates[name] = parents[name].copy()
        candidates[name][active[name]] = np.clip(
            parents[name][active[name]] + directions[name][active[name]],
            0.001,
            0.999,
        )
        metrics[name] = paired_metrics(
            _axes(frame), parents[name], candidates[name], active[name]
        )

    family = [parents["full_2024"], candidates["full_2024"]]
    robustness = _robustness(
        _axes(frames["full_2024"]),
        parents["full_2024"],
        candidates["full_2024"],
        active["full_2024"],
        family,
    )
    increment = candidates["full_2024"] - parents["full_2024"]
    support = (np.abs(increment) > 0.0) | (np.abs(v320_increment) > 0.0)
    correlation = (
        float(np.corrcoef(increment[support], v320_increment[support])[0, 1])
        if support.sum() > 2 and np.std(increment[support]) > 0 and np.std(v320_increment[support]) > 0
        else 0.0
    )
    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0
        and metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2022"]["positive_month_fraction"] >= 5.0 / 7.0
        and metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
    )
    locked_pass = bool(
        source_pass
        and metrics["full_2024"]["gain"] >= 1.0
        and metrics["full_2024"]["positive_month_fraction"] >= 0.75
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
        **{f"active_{name}": active[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "frozen_recipe": {
            "source": "v221 role_weight=0.50 selected candidate",
            "embedded_joint_h1_screen_scale": 0.10,
            "new_dose": 1.0,
            "route": "R_CORE",
        },
        "metrics": metrics,
        "locked_whole_rms_shift": float(np.sqrt(np.mean(np.square(increment)))),
        "locked_robustness": robustness,
        "direction_correlation_with_v320_increment": correlation,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "v221_direction_frozen": True,
            "role_or_dose_retuned": False,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
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
    parser.add_argument("--v221-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, args.v221_axes, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
