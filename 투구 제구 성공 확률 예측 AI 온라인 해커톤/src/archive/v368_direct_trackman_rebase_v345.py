"""Rebase the frozen v260 direct TrackMan command+batter delta above v345.

The v260 expert was selected without v345 and combines a source-stable
command-dispersion specialist with a disjoint batter-exposure complement.
Its already frozen probability delta is added unchanged above v345.  The two
source axes must remain positive before the full-2024 axis is evaluated.
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
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V368_DIRECT_TRACKMAN_REBASE_V345_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def run(train_csv: Path, v260_axes: Path, v335_axes: Path, v345_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[train["season"].eq(2023) & train["game_month"].ge(8)].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v260_axes, allow_pickle=False) as saved:
        direction = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            - saved[f"parent_{name}"].astype(np.float64)
            for name in ORIGINS
        }
        historical_active = {
            name: saved[f"active_{name}"].astype(bool) for name in ORIGINS
        }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = saved["candidate_full_2024"].astype(np.float64) - saved["parent_full_2024"].astype(np.float64)

    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name in ("full_2022", "late_2023"):
        active[name] = historical_active[name] & (np.abs(direction[name]) > 1e-15)
        candidates[name] = np.clip(parents[name] + direction[name], 0.001, 0.999)
        metrics[name] = axis_metrics(frames[name], parents[name], candidates[name], active[name])
        metrics[name]["full_row_rms_shift"] = full_row_rms(parents[name], candidates[name])
    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0
        and metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2022"]["positive_month_fraction"] >= 2.0 / 3.0
        and metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
        and metrics["full_2022"]["worst_month_gain"] > -5.0
        and metrics["late_2023"]["worst_month_gain"] > -5.0
    )
    restrictions = {
        "official_train_and_trackman_only": True,
        "frozen_v260_command_batter_delta": True,
        "no_route_weight_or_model_retuning": True,
        "full_2024_opened_only_after_source_gate": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not source_pass:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_metrics_vs_v345": metrics,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        return summary

    name = "full_2024"
    active[name] = historical_active[name] & (np.abs(direction[name]) > 1e-15)
    candidates[name] = np.clip(parents[name] + direction[name], 0.001, 0.999)
    locked = axis_metrics(frames[name], parents[name], candidates[name], active[name])
    locked["full_row_rms_shift"] = full_row_rms(parents[name], candidates[name])
    union = active[name] | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(direction[name][union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames[name][TARGET].to_numpy(np.float64),
        "game_month": frames[name]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames[name]["pitcher_id"].to_numpy(),
        "batter_id": frames[name]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames[name]), dtype=bool),
    }
    robustness = _robustness(axes24, parents[name], candidates[name], active[name], [parents[name], candidates[name]])
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.5
        and locked["worst_month_gain"] > -15.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{f"candidate_{origin}": candidates[origin] for origin in ORIGINS},
        **{f"active_{origin}": active[origin] for origin in ORIGINS},
        **{f"direction_{origin}": direction[origin] for origin in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "frozen_recipe": {"source": "v260 command+batter", "dose": 1.0, "rebase_parent": "v345"},
        "source_metrics_vs_v345": metrics,
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v260-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v260_axes, args.v335_axes, args.v345_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
