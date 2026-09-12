"""Single-origin unanimous baseball-expert router above the Public v345 parent.

v330 showed that pooled two-origin cell selection was unstable.  This audit is
stricter: an expert is assigned to a cell only when routers fitted separately
on every available source origin choose the same expert.  Leave-one-origin
predictions use only the two remaining origins.  The final 2025-oriented
router requires unanimity across full-2022, late-2023 and full-2024.

The selected router replaces v345's player-transition component inside the
assigned R_CORE cells.  Existing workload and Beta components are retained,
except when the selected expert is that same component, preventing double
application.  F and R_ANCHOR reproduce v345 exactly.
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
    fit_router,
    schema_key,
)


PROTOCOL = "V359_SINGLE_ORIGIN_UNANIMOUS_ROUTER_V1"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def unanimous_router(routers: list[dict[str, str]]) -> dict[str, str]:
    """Keep only cell assignments present and identical in every router."""

    if not routers:
        return {}
    groups = set(routers[0])
    for router in routers[1:]:
        groups.intersection_update(router)
    return {
        group: routers[0][group]
        for group in sorted(groups)
        if all(router[group] == routers[0][group] for router in routers[1:])
    }


def _load_parents(
    v285_axes: Path, v318_axes: Path
) -> dict[str, np.ndarray]:
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64)
        }
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
    return parents


def _v345_replacement(
    frame: pd.DataFrame,
    v345: np.ndarray,
    v335: np.ndarray,
    directions: dict[str, np.ndarray],
    keys: np.ndarray,
    router: dict[str, str],
    player_increment: np.ndarray,
    workload_increment: np.ndarray,
    beta_increment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace the v345 player expert without duplicating retained experts."""

    candidate = np.asarray(v345, dtype=np.float64).copy()
    assignment = np.full(len(frame), "PARENT", dtype="<U32")
    archetypes = build_archetypes(frame)
    rcore = archetypes["route"].eq("R_CORE").to_numpy()
    for group, expert in router.items():
        selected = rcore & (keys == group)
        if not selected.any():
            continue
        replacement = directions[expert] - player_increment
        if expert == "joint_role_h1":
            replacement = replacement - workload_increment
        elif expert == "beta_binomial":
            replacement = replacement - beta_increment
        candidate[selected] = np.clip(
            v345[selected] + replacement[selected], 0.001, 0.999
        )
        assignment[selected] = expert
    active = assignment != "PARENT"
    # The proven v345 protected routes must remain byte-for-byte identical.
    protected = ~rcore
    if not np.array_equal(candidate[protected], v345[protected]):
        raise ValueError("F/R_ANCHOR protection failure")
    if not np.allclose(v335, v345 - player_increment - workload_increment - beta_increment):
        raise ValueError("v345 component decomposition failure")
    return candidate, active, assignment


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
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
    parents = _load_parents(v285_axes, v318_axes)
    directions = {
        origin: {
            expert: _candidate_minus_parent(path, origin)
            for expert, path in expert_paths.items()
        }
        for origin in ORIGINS
    }
    archetypes = {origin: build_archetypes(frames[origin]) for origin in ORIGINS}

    trials: list[dict[str, Any]] = []
    final_routers: dict[tuple[str, ...], dict[str, str]] = {}
    router_rows: list[dict[str, str]] = []
    for schema in SCHEMAS:
        label = "+".join(schema)
        keys = {origin: schema_key(archetypes[origin], schema) for origin in ORIGINS}
        single: dict[str, dict[str, str]] = {}
        for origin in ORIGINS:
            single[origin], _ = fit_router(
                frames[origin], parents[origin], directions[origin], keys[origin]
            )
            for group, expert in single[origin].items():
                router_rows.append(
                    {"schema": label, "fit_origin": origin, "group": group, "expert": expert}
                )

        loo_metrics: dict[str, Any] = {}
        loo_cells: dict[str, int] = {}
        for held in ORIGINS:
            fitted = [origin for origin in ORIGINS if origin != held]
            router = unanimous_router([single[origin] for origin in fitted])
            regular = frames[held]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[held], directions[held], keys[held], router, regular
            )
            loo_metrics[held] = axis_metrics(
                frames[held], parents[held], candidate, active
            )
            loo_cells[held] = len(router)

        final_router = unanimous_router([single[origin] for origin in ORIGINS])
        final_routers[schema] = final_router
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
                "loo_cells": loo_cells,
                "minimum_loo_gain": float(min(loo_gains)),
                "mean_loo_gain": float(np.mean(loo_gains)),
                "minimum_final_gain": float(min(final_gains)),
                "loo_metrics": loo_metrics,
                "final_metrics": final_metrics,
            }
        )

    pd.DataFrame(router_rows).to_csv(output_dir / "single_origin_routers.csv", index=False)
    eligible_trials = [trial for trial in trials if trial["eligible"]]
    nonempty_trials = [trial for trial in trials if trial["final_cells"] > 0]
    pool = eligible_trials if eligible_trials else (nonempty_trials or trials)
    selected = max(
        pool,
        key=lambda trial: (
            trial["eligible"], trial["minimum_loo_gain"],
            trial["mean_loo_gain"], trial["minimum_final_gain"],
            -trial["complexity"],
        ),
    )
    selected_schema = next(
        schema for schema in SCHEMAS if "+".join(schema) == selected["schema"]
    )
    final_router = final_routers[selected_schema]
    pd.DataFrame(
        [{"group": group, "expert": expert} for group, expert in sorted(final_router.items())]
    ).to_csv(output_dir / "final_router.csv", index=False)

    with np.load(v335_axes, allow_pickle=False) as saved:
        v335 = saved["candidate_full_2024"].astype(np.float64)
    with np.load(v345_axes, allow_pickle=False) as saved:
        v345_parent = saved["parent_full_2024"].astype(np.float64)
        v345 = saved["candidate_full_2024"].astype(np.float64)
        v343_increment = saved["v343_increment_full_2024"].astype(np.float64)
        beta_increment = saved["beta_increment_full_2024"].astype(np.float64)
    if not np.array_equal(v345_parent, v335):
        raise ValueError("v345/v335 parent mismatch")
    player_increment = directions["full_2024"]["player_transition"]
    workload_increment = v343_increment - player_increment
    full24_keys = schema_key(archetypes["full_2024"], selected_schema)
    rebased, active24, assignment24 = _v345_replacement(
        frames["full_2024"], v345, v335, directions["full_2024"],
        full24_keys, final_router, player_increment, workload_increment,
        beta_increment,
    )
    rebased_metrics = axis_metrics(frames["full_2024"], v345, rebased, active24)
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = (
        _robustness(axes24, v345, rebased, active24, [v345, rebased])
        if active24.any()
        else {"not_run": "unanimous final router has no active R_CORE rows"}
    )
    eligible = bool(
        selected["eligible"]
        and active24.any()
        and rebased_metrics["gain"] >= 2.0
        and rebased_metrics["positive_month_fraction"] >= 0.625
        and robustness.get("pitcher", {}).get("p05", -np.inf) > 0.0
        and robustness.get("chronological_block", {}).get("p05", -np.inf) > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=v345,
        candidate_full_2024=rebased,
        active_full_2024=active24,
        assignment_full_2024=assignment24,
        player_increment_full_2024=player_increment,
        workload_increment_full_2024=workload_increment,
        beta_increment_full_2024=beta_increment,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if eligible else (
            "locked_reject" if selected["eligible"] else "cross_origin_reject"
        ),
        "selected_schema": selected,
        "schema_trials": trials,
        "final_router": final_router,
        "v345_replacement_metrics": rebased_metrics,
        "v345_replacement_robustness": robustness,
        "assignment_counts": {
            expert: int(np.sum(assignment24 == expert))
            for expert in ("PARENT", *EXPERT_ORDER)
            if np.any(assignment24 == expert)
        },
        "eligible_for_packaging": eligible,
        "restrictions": {
            "organizer_data_only": True,
            "leave_one_origin_router_uses_only_other_origins": True,
            "final_cell_requires_three_single_origin_routers_to_agree": True,
            "schema_selected_by_leave_one_origin_metrics": True,
            "v345_f_and_r_anchor_preserved_exactly": True,
            "duplicate_workload_or_beta_application_prevented": True,
            "evaluation_row_order_ids_or_aggregates_used": False,
            "public_score_used_for_selection": False,
            "full_2024_is_training_origin_not_fresh_holdout": True,
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
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    for name in EXPERT_ORDER:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expert_paths = {name: getattr(args, name) for name in EXPERT_ORDER}
    result = run(
        args.train_csv, args.v285_axes, args.v318_axes, args.v335_axes,
        args.v345_axes, expert_paths, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
