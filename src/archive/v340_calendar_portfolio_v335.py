"""Add the frozen v261 R_CORE calendar expert to the v339 portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V340_CALENDAR_PORTFOLIO_V335_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def add_component(
    parent: np.ndarray,
    old_parent: np.ndarray,
    old_candidate: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    delta = np.asarray(old_candidate, dtype=np.float64) - np.asarray(
        old_parent, dtype=np.float64
    )
    output[active] = np.clip(output[active] + delta[active], 0.001, 0.999)
    return output


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        parent, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def run(
    train_csv: Path,
    v335_axes: Path,
    v339_axes: Path,
    v261_axes: Path,
    v261_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_id", TARGET],
        low_memory=False,
    )
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
    with np.load(v261_axes, allow_pickle=False) as saved:
        calendar_old_parent = {
            origin: saved[f"parent_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        calendar_old_candidate = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        calendar_active = {
            origin: saved[f"active_{origin}"].astype(bool)
            for origin in ORIGINS
        }
    with np.load(v339_axes, allow_pickle=False) as saved:
        v339_parent = saved["parent_full_2024"].astype(np.float64)
        v339_candidate = saved["candidate_full_2024"].astype(np.float64)
        v339_active = saved["active_full_2024"].astype(bool)
    if np.max(np.abs(v339_parent - parents["full_2024"])) > 1e-12:
        raise ValueError("v339 parent is not v335")

    calendar_candidates: dict[str, np.ndarray] = {}
    calendar_metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        calendar_candidates[origin] = add_component(
            parents[origin],
            calendar_old_parent[origin],
            calendar_old_candidate[origin],
            calendar_active[origin],
        )
        calendar_metrics[origin] = axis_metrics(
            frames[origin],
            parents[origin],
            calendar_candidates[origin],
            calendar_active[origin],
        )
        calendar_metrics[origin]["full_row_rms_shift"] = full_row_rms(
            parents[origin], calendar_candidates[origin]
        )

    portfolio = add_component(
        v339_candidate,
        calendar_old_parent["full_2024"],
        calendar_old_candidate["full_2024"],
        calendar_active["full_2024"],
    )
    portfolio_active = v339_active | calendar_active["full_2024"]
    frame24 = frames["full_2024"]
    portfolio_metrics = axis_metrics(
        frame24, parents["full_2024"], portfolio, portfolio_active
    )
    portfolio_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], portfolio
    )
    v339_metrics = axis_metrics(
        frame24, parents["full_2024"], v339_candidate, v339_active
    )
    v339_metrics["full_row_rms_shift"] = full_row_rms(
        parents["full_2024"], v339_candidate
    )

    v339_delta = v339_candidate - parents["full_2024"]
    calendar_delta = calendar_candidates["full_2024"] - parents["full_2024"]
    union = (np.abs(v339_delta) > 1e-15) | (np.abs(calendar_delta) > 1e-15)
    correlation = float(np.corrcoef(v339_delta[union], calendar_delta[union])[0, 1])
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
        portfolio,
        portfolio_active,
        [
            parents["full_2024"],
            v339_candidate,
            calendar_candidates["full_2024"],
            portfolio,
        ],
    )
    passed = bool(
        portfolio_metrics["gain"] > v339_metrics["gain"]
        and portfolio_metrics["gain"] >= 3.0
        and portfolio_metrics["positive_month_fraction"] >= 0.75
        and all(calendar_metrics[name]["gain"] > 0.0 for name in ORIGINS)
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=portfolio,
        active_full_2024=portfolio_active,
        calendar_increment_full_2024=calendar_delta,
        v339_increment_full_2024=v339_delta,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "inventory_candidate" if passed else "inventory_reject",
        "recipe": {
            "parent": "v335",
            "portfolio_parent": "v339 transition + three-seed workload",
            "new_component": "v261 frozen command+batter R_CORE, months 4-9",
            "route_or_calendar_retuned": False,
        },
        "calendar_rebase_metrics": calendar_metrics,
        "v339_metrics": v339_metrics,
        "portfolio_metrics": portfolio_metrics,
        "increment_correlation": correlation,
        "support": {
            "v339_rows": int(v339_active.sum()),
            "calendar_rows": int(calendar_active["full_2024"].sum()),
            "overlap_rows": int(
                np.count_nonzero(v339_active & calendar_active["full_2024"])
            ),
        },
        "locked_robustness": robustness,
        "rms_goal_context": {
            "locked_full_row_rms": portfolio_metrics["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": float(
                portfolio_metrics["full_row_rms_shift"] / 0.004551
            ),
        },
        "candidate_gate_passed": passed,
        "historical_v261": json.loads(
            v261_summary.read_text(encoding="utf-8")
        )["restrictions"],
        "selection_warning": (
            "full-2024 and the calendar rule were already exposed in earlier "
            "research; this is an inventory rebase, not a fresh holdout"
        ),
        "restrictions": {
            "official_train_only": True,
            "v261_formula_and_calendar_frozen": True,
            "v339_components_frozen": True,
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
    parser.add_argument("--v261-axes", type=Path, required=True)
    parser.add_argument("--v261-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v339_axes,
        args.v261_axes,
        args.v261_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
