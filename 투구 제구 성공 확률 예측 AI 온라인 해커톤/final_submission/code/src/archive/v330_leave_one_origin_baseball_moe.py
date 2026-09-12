"""Leave-one-origin audit and final three-origin baseball expert router.

For each baseball schema, a router is built from two origins and evaluated on
the third.  The three held origins are full-2022, late-2023 and full-2024.  A
schema is eligible only when every held-origin gain is positive.  The final
2025-oriented router then keeps cell/expert pairs that improve total squared
error and a majority of months in all three origins.  This uses all official
training labels for the final model while retaining honest origin-level
cross-validation for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import (
    EXPERT_ORDER,
    SCHEMAS,
    TARGET,
    _candidate_minus_parent,
    apply_router,
    axis_metrics,
    build_archetypes,
    schema_key,
)
from src.archive.v329_two_origin_stable_expert_cells import stable_router


PROTOCOL = "V330_LEAVE_ONE_ORIGIN_BASEBALL_MOE_V1"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    expert_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
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
    archetypes = {origin: build_archetypes(frames[origin]) for origin in ORIGINS}
    directions = {
        origin: {
            expert: _candidate_minus_parent(path, origin)
            for expert, path in expert_paths.items()
        }
        for origin in ORIGINS
    }

    trials: list[dict[str, Any]] = []
    final_routers: dict[tuple[str, ...], dict[str, str]] = {}
    evidence_frames: list[pd.DataFrame] = []
    loo_router_rows: list[dict[str, Any]] = []
    for schema in SCHEMAS:
        label = "+".join(schema)
        keys = {origin: schema_key(archetypes[origin], schema) for origin in ORIGINS}
        loo_metrics: dict[str, Any] = {}
        for held in ORIGINS:
            fitted = tuple(origin for origin in ORIGINS if origin != held)
            router, evidence = stable_router(
                frames, parents, directions, keys, sources=fitted
            )
            for group, expert in router.items():
                loo_router_rows.append(
                    {
                        "schema": label,
                        "held_origin": held,
                        "fit_origins": "+".join(fitted),
                        "group": group,
                        "expert": expert,
                    }
                )
            regular = frames[held]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[held], directions[held], keys[held], router, regular
            )
            loo_metrics[held] = axis_metrics(
                frames[held], parents[held], candidate, active
            )

        final_router, evidence = stable_router(
            frames, parents, directions, keys, sources=ORIGINS
        )
        final_routers[schema] = final_router
        evidence.insert(0, "schema", label)
        evidence_frames.append(evidence)
        final_metrics: dict[str, Any] = {}
        for origin in ORIGINS:
            regular = frames[origin]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[origin], directions[origin], keys[origin], final_router, regular
            )
            final_metrics[origin] = axis_metrics(
                frames[origin], parents[origin], candidate, active
            )
        loo_gains = [loo_metrics[origin]["gain"] for origin in ORIGINS]
        final_gains = [final_metrics[origin]["gain"] for origin in ORIGINS]
        eligible = bool(
            final_router
            and min(loo_gains) > 0.0
            and min(final_gains) > 0.0
            and all(
                loo_metrics[origin]["positive_month_fraction"] >= 0.5
                for origin in ORIGINS
            )
        )
        trials.append(
            {
                "schema": label,
                "complexity": len(schema),
                "eligible": eligible,
                "final_cells": len(final_router),
                "minimum_loo_gain": float(min(loo_gains)),
                "mean_loo_gain": float(np.mean(loo_gains)),
                "minimum_final_gain": float(min(final_gains)),
                "loo_metrics": loo_metrics,
                "final_metrics": final_metrics,
            }
        )
    pd.DataFrame(loo_router_rows).to_csv(
        output_dir / "leave_one_origin_routers.csv", index=False
    )
    pd.concat(evidence_frames, ignore_index=True).to_csv(
        output_dir / "three_origin_cell_evidence.csv", index=False
    )
    eligible_trials = [trial for trial in trials if trial["eligible"]]
    pool = eligible_trials if eligible_trials else trials
    selected = max(
        pool,
        key=lambda trial: (
            trial["eligible"], trial["minimum_loo_gain"],
            trial["minimum_final_gain"], trial["mean_loo_gain"],
            -trial["complexity"],
        ),
    )
    selected_schema = next(
        schema for schema in SCHEMAS if "+".join(schema) == selected["schema"]
    )
    router = final_routers[selected_schema]
    pd.DataFrame(
        [{"group": group, "expert": expert} for group, expert in sorted(router.items())]
    ).to_csv(output_dir / "final_router.csv", index=False)

    candidates: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    assignments: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        keys = schema_key(archetypes[origin], selected_schema)
        regular = frames[origin]["game_type"].astype(str).eq("R").to_numpy()
        candidates[origin], actives[origin], assignments[origin] = apply_router(
            parents[origin], directions[origin], keys, router, regular
        )
        metrics[origin] = axis_metrics(
            frames[origin], parents[origin], candidates[origin], actives[origin]
        )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness24 = _robustness(
        axes24, parents["full_2024"], candidates["full_2024"],
        actives["full_2024"], [parents["full_2024"], candidates["full_2024"]],
    )
    promote = bool(
        selected["eligible"]
        and robustness24["pitcher"]["p05"] > 0.0
        and robustness24["chronological_block"]["p05"] > 0.0
        and robustness24["reality_check"]["p_value"] < 0.10
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{f"candidate_{origin}": candidates[origin] for origin in ORIGINS},
        **{f"active_{origin}": actives[origin] for origin in ORIGINS},
        **{f"assignment_{origin}": assignments[origin] for origin in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "cross_origin_candidate" if promote else "cross_origin_reject",
        "selected_schema": selected,
        "schema_trials": trials,
        "final_router": router,
        "metrics": metrics,
        "full_2024_training_axis_robustness": robustness24,
        "cross_origin_gate_passed": bool(selected["eligible"]),
        "promotion_gate_passed": promote,
        "assignment_counts": {
            origin: {
                expert: int(np.sum(assignments[origin] == expert))
                for expert in ("PARENT", *EXPERT_ORDER)
                if np.any(assignments[origin] == expert)
            }
            for origin in ORIGINS
        },
        "restrictions": {
            "organizer_data_only": True,
            "all_three_origins_used_for_final_2025_router": True,
            "model_selection_uses_leave_one_origin_out_gain": True,
            "f_route_protected_as_v320": True,
            "evaluation_row_order_ids_or_aggregates_used": False,
            "public_score_used_for_selection": False,
            "full_2024_is_final_training_axis_not_locked_holdout": True,
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
    for name in EXPERT_ORDER:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expert_paths = {name: getattr(args, name) for name in EXPERT_ORDER}
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, expert_paths, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
