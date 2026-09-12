"""Reliability-routed recent-F dose above the frozen v320 portfolio.

The recent Futures direct expert is the only newly added model family with
positive Public evidence.  v320 applies it at a uniform 20% dose.  This audit
asks whether fixed baseball/reliability cells support either 10% or 30% while
leaving all other F rows at 20%.

Cell actions and the routing schema are selected only on late-2023 F rows and
must improve both source months.  Full-2024 is a locked transfer audit.  Every
declared schema is included in the locked White Reality Check.
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
from src.archive.v328_baseball_archetype_consensus_moe import TARGET, build_archetypes


PROTOCOL = "V332_FUTURES_RELIABILITY_DOSE_ROUTER_V1"
AXES = ("late_2023", "full_2024")
MIN_CELL_ROWS = 300
MIN_MONTH_ROWS = 100
ACTIONS = (-1, 1)  # one frozen 10%-dose step below/above v320's 20%
SCHEMAS = (
    ("support",),
    ("count",),
    ("pitchmix",),
    ("platoon_state",),
    ("leverage",),
    ("parent_bin",),
    ("direct_direction",),
    ("direct_magnitude",),
    ("support", "count"),
    ("support", "pitchmix"),
    ("support", "direct_direction"),
    ("support", "direct_magnitude"),
    ("parent_bin", "direct_direction"),
    ("direct_direction", "direct_magnitude"),
)


def reliability_cells(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direct_step: np.ndarray,
) -> pd.DataFrame:
    """Fixed row-local baseball and prediction-reliability categories."""

    cells = build_archetypes(frame).copy()
    parent = np.asarray(parent, dtype=np.float64)
    direct_step = np.asarray(direct_step, dtype=np.float64)
    cells["parent_bin"] = pd.cut(
        parent,
        [-np.inf, 0.45, 0.50, 0.55, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
        right=False,
    ).astype(str)
    cells["direct_direction"] = np.where(direct_step >= 0.0, "UP", "DOWN")
    magnitude = np.abs(direct_step)
    cells["direct_magnitude"] = pd.cut(
        magnitude,
        [-np.inf, 0.0025, 0.0050, np.inf],
        labels=["SMALL", "MEDIUM", "LARGE"],
        right=False,
    ).astype(str)
    return cells


def schema_key(cells: pd.DataFrame, schema: tuple[str, ...]) -> np.ndarray:
    return cells.loc[:, list(schema)].astype(str).agg("|".join, axis=1).to_numpy()


def sse_gain(
    target: np.ndarray,
    parent: np.ndarray,
    direct_step: np.ndarray,
    mask: np.ndarray,
    action: int,
) -> float:
    candidate = np.clip(
        parent[mask] + int(action) * direct_step[mask], 0.001, 0.999
    )
    return float(
        np.sum(np.square(target[mask] - parent[mask]))
        - np.sum(np.square(target[mask] - candidate))
    )


def fit_router(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direct_step: np.ndarray,
    keys: np.ndarray,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Choose +/- one dose step only when both source months improve."""

    target = frame[TARGET].to_numpy(np.float64)
    months = frame["game_month"].to_numpy(np.int16)
    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    router: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for group in sorted(np.unique(keys[futures])):
        selected = futures & (keys == group)
        if int(selected.sum()) < MIN_CELL_ROWS:
            continue
        source_months = sorted(np.unique(months[selected]).tolist())
        best: tuple[float, int] | None = None
        for action in ACTIONS:
            month_gains: dict[int, float] = {}
            month_support: dict[int, int] = {}
            for month in source_months:
                mask = selected & (months == month)
                month_support[int(month)] = int(mask.sum())
                month_gains[int(month)] = sse_gain(
                    target, parent, direct_step, mask, action
                )
            total_gain = sse_gain(target, parent, direct_step, selected, action)
            eligible = bool(
                total_gain > 0.0
                and len(source_months) >= 2
                and all(month_support[int(month)] >= MIN_MONTH_ROWS for month in source_months)
                and all(month_gains[int(month)] > 0.0 for month in source_months)
            )
            rows.append(
                {
                    "group": group,
                    "rows": int(selected.sum()),
                    "action": int(action),
                    "total_sse_gain": total_gain,
                    "eligible": eligible,
                    "month_support": json.dumps(month_support, sort_keys=True),
                    "month_sse_gain": json.dumps(month_gains, sort_keys=True),
                }
            )
            if eligible and (best is None or total_gain > best[0]):
                best = (total_gain, int(action))
        if best is not None:
            router[group] = best[1]
    return router, pd.DataFrame(rows)


def apply_router(
    parent: np.ndarray,
    direct_step: np.ndarray,
    keys: np.ndarray,
    router: dict[str, int],
    futures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.zeros(len(output), dtype=bool)
    action = np.zeros(len(output), dtype=np.int8)
    for group, value in router.items():
        selected = np.asarray(futures, dtype=bool) & (keys == group)
        output[selected] = np.clip(
            parent[selected] + int(value) * direct_step[selected], 0.001, 0.999
        )
        active |= selected
        action[selected] = int(value)
    return output, active, action


def _axis(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }


def run(train_csv: Path, v318_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "li", "asof_pitcher_n",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) for axis in AXES
        }
        direct = {
            axis: saved[f"direct_direction_{axis}"].astype(np.float64) for axis in AXES
        }
    cells = {
        axis: reliability_cells(frames[axis], parents[axis], direct[axis]) for axis in AXES
    }
    futures = {
        axis: frames[axis]["game_type"].astype(str).eq("F").to_numpy() for axis in AXES
    }
    axes = {axis: _axis(frames[axis]) for axis in AXES}

    trials: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    routers: dict[str, dict[str, int]] = {}
    evidence: list[pd.DataFrame] = []
    for schema in SCHEMAS:
        label = "+".join(schema)
        source_keys = schema_key(cells["late_2023"], schema)
        router, table = fit_router(
            frames["late_2023"], parents["late_2023"], direct["late_2023"], source_keys
        )
        routers[label] = router
        table.insert(0, "schema", label)
        evidence.append(table)
        per_axis: dict[str, Any] = {}
        for axis in AXES:
            keys = schema_key(cells[axis], schema)
            candidate, active, action = apply_router(
                parents[axis], direct[axis], keys, router, futures[axis]
            )
            per_axis[axis] = paired_metrics(
                axes[axis], parents[axis], candidate, active
            ) if active.any() else {
                "gain": 0.0, "active_rows": 0, "positive_month_fraction": 0.0,
                "worst_month_gain": 0.0, "months": [],
            }
            if axis == "full_2024":
                predictions[label] = candidate
                actives[label] = active
                actions[label] = action
        source = per_axis["late_2023"]
        source_pass = bool(
            router
            and source["gain"] > 0.0
            and source["positive_month_fraction"] == 1.0
        )
        trials.append(
            {
                "schema": label,
                "complexity": len(schema),
                "cells": len(router),
                "source_passed": source_pass,
                "metrics": per_axis,
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            row["source_passed"], row["metrics"]["late_2023"]["gain"],
            -row["complexity"],
        ),
    )
    selected_label = str(selected["schema"])
    selected_candidate = predictions[selected_label]
    selected_active = actives[selected_label]
    robustness = _robustness(
        axes["full_2024"], parents["full_2024"], selected_candidate,
        selected_active,
        [parents["full_2024"], *[predictions["+".join(schema)] for schema in SCHEMAS]],
    )
    locked = selected["metrics"]["full_2024"]
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    pd.concat(evidence, ignore_index=True).to_csv(
        output_dir / "source_cell_evidence.csv", index=False
    )
    pd.DataFrame(
        [
            {"group": group, "action": action, "resulting_direct_dose": 0.20 + 0.10 * action}
            for group, action in sorted(routers[selected_label].items())
        ]
    ).to_csv(output_dir / "selected_router.csv", index=False)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{axis}": parents[axis] for axis in AXES},
        candidate_full_2024=selected_candidate,
        active_full_2024=selected_active,
        action_full_2024=actions[selected_label],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_candidate" if selected["source_passed"] and locked_pass and robust_pass else "locked_reject",
        "selected": selected,
        "selected_router": routers[selected_label],
        "schema_trials": trials,
        "locked_robustness": robustness,
        "source_gate_passed": bool(selected["source_passed"]),
        "locked_gate_passed": locked_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(selected["source_passed"] and locked_pass and robust_pass),
        "restrictions": {
            "official_train_only": True,
            "v320_is_parent": True,
            "late_2023_only_selects_schema_and_cells": True,
            "both_source_months_must_improve_per_cell": True,
            "full_2024_locked_from_selection": True,
            "all_declared_schemas_in_reality_check": True,
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
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.train_csv, args.v318_axes, args.output_dir),
        ensure_ascii=False, indent=2, default=float,
    ))


if __name__ == "__main__":
    main()
