"""Audit a mature-current-season hierarchy above v345 on three origins.

The raw hierarchy and 0.20 R_ANCHOR blend were selected in the archived v39
study.  This experiment adds one baseball-motivated deployment restriction:
use the current-season posterior only from August, when the season sample is
mature.  The same month gate and formula are applied to full-2022, late-2023,
and the locked full-2024 v345 axis.  Every query uses only its row's ASOF
sufficient statistics and frozen prior-season tables.
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
from src.archive.v39_hierarchical_season_forecast import forecast_bank
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V354_LATE_HIERARCHY_REBASE_V345_V1"
SIGNAL = "hier::domain_latest_k80_p75"
WEIGHT = 0.20
START_MONTH = 8
TARGET = "control_success"


def late_anchor_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    mature = pd.to_numeric(frame["game_month"], errors="coerce").fillna(0).ge(
        START_MONTH
    ).to_numpy()
    return anchor & mature


def blend_toward_hierarchy(
    frame: pd.DataFrame,
    parent: np.ndarray,
    hierarchy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    hierarchy = np.asarray(hierarchy, dtype=np.float64)
    if len(frame) != len(parent) or len(parent) != len(hierarchy):
        raise ValueError("hierarchy axis length mismatch")
    active = late_anchor_mask(frame)
    candidate = parent.copy()
    candidate[active] = np.clip(
        (1.0 - WEIGHT) * parent[active] + WEIGHT * hierarchy[active],
        0.001,
        0.999,
    )
    return candidate, active


def _full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
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
    v345_axes: Path,
    v352_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(train_csv, low_memory=False))
    full_frames = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    frames = {
        "full_2022": full_frames[2022],
        "late_2023": full_frames[2023].loc[
            full_frames[2023]["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": full_frames[2024],
    }
    banks = {
        year: forecast_bank(train, year)[SIGNAL]
        for year in (2022, 2023, 2024)
    }
    hierarchy = {
        "full_2022": banks[2022],
        "late_2023": banks[2023][
            full_frames[2023]["game_month"].ge(8).to_numpy()
        ],
        "full_2024": banks[2024],
    }
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64),
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        v335_2024 = saved["parent_full_2024"].astype(np.float64)
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)

    metrics: dict[str, dict[str, Any]] = {}
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for name in ("full_2022", "late_2023", "full_2024"):
        candidate, mask = blend_toward_hierarchy(
            frames[name], parents[name], hierarchy[name]
        )
        result = axis_metrics(frames[name], parents[name], candidate, mask)
        result["full_row_rms_shift"] = _full_row_rms(parents[name], candidate)
        metrics[name] = result
        candidates[name] = candidate
        active[name] = mask

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
    with np.load(v352_axes, allow_pickle=False) as saved:
        pfd_late = saved["increment_late_2024"].astype(np.float64)
        pfd_active_late = saved["active_late_2024"].astype(bool)
    late24 = frame24["game_month"].ge(8).to_numpy()
    hierarchy_increment = candidates["full_2024"] - parents["full_2024"]
    overlap = active["full_2024"][late24] & pfd_active_late
    correlation_pfd = float(
        np.corrcoef(hierarchy_increment[late24][overlap], pfd_late[overlap])[0, 1]
    )
    # v345's post-v335 increments are R_CORE-only, so the mature R_ANCHOR
    # hierarchy is support-disjoint by construction.
    overlap_v345 = int(
        np.count_nonzero(
            active["full_2024"]
            & (np.abs(parents["full_2024"] - v335_2024) > 1e-15)
        )
    )
    passed = bool(
        all(metrics[name]["gain"] > 0.0 for name in metrics)
        and all(metrics[name]["positive_month_fraction"] == 1.0 for name in metrics)
        and all(metrics[name]["worst_month_gain"] > 0.0 for name in metrics)
        and metrics["full_2024"]["gain"] >= 3.0
        and abs(correlation_pfd) <= 0.30
        and overlap_v345 == 0
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )

    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=candidates["full_2024"],
        hierarchy_full_2024=hierarchy["full_2024"],
        hierarchy_increment_full_2024=hierarchy_increment,
        active_full_2024=active["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "reject",
        "recipe": {
            "parent": "v345 / Public 1182.94969702",
            "signal": SIGNAL,
            "domain": "R_ANCHOR",
            "start_month": START_MONTH,
            "weight_toward_hierarchy": WEIGHT,
            "route_or_weight_retuned": False,
        },
        "metrics": metrics,
        "locked_robustness": robustness,
        "diversity": {
            "correlation_with_v352_pfd_on_overlap": correlation_pfd,
            "overlap_rows_with_post_v335_v345_increment": overlap_v345,
        },
        "candidate_gate_passed": passed,
        "selection_warning": (
            "The raw hierarchy and 0.20 R_ANCHOR dose were selected in v39. "
            "The August maturity gate is a new baseball reliability hypothesis; "
            "2022/late-2023 are supporting source axes and full-2024 is the "
            "locked v345 rebase axis."
        ),
        "restrictions": {
            "official_train_only": True,
            "row_local_asof_counts_only": True,
            "frozen_prior_season_tables": True,
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
    parser.add_argument("--v352-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.v335_axes,
                args.v345_axes,
                args.v352_axes,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
