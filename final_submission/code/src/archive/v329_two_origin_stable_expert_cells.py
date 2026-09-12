"""Route frozen expert directions only where both source origins agree.

Unlike v328's winner-identity consensus, this protocol evaluates every frozen
expert independently inside each baseball cell.  A cell/expert pair survives
only when it improves total squared error and a majority of months in both
full-2022 and late-2023.  Among survivors in a cell, the maximin per-row effect
selects one expert.  Schema selection is source-only; full-2024 is evaluated
after the router is frozen.  The v320 Futures route remains untouched.
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
    MIN_GROUP_ROWS,
    SCHEMAS,
    TARGET,
    _candidate_minus_parent,
    _sse_gain,
    apply_router,
    axis_metrics,
    build_archetypes,
    schema_key,
)


PROTOCOL = "V329_TWO_ORIGIN_STABLE_EXPERT_CELLS_V1"


def stable_router(
    frames: dict[str, pd.DataFrame],
    parents: dict[str, np.ndarray],
    directions: dict[str, dict[str, np.ndarray]],
    keys: dict[str, np.ndarray],
    sources: tuple[str, ...] = ("full_2022", "late_2023"),
) -> tuple[dict[str, str], pd.DataFrame]:
    if len(sources) < 2:
        raise ValueError("stable routing requires at least two origins")
    common_groups = set(np.unique(keys[sources[0]]))
    for source in sources[1:]:
        common_groups.intersection_update(np.unique(keys[source]))
    table: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for group in sorted(common_groups):
        evidence: list[dict[str, Any]] = []
        for expert in EXPERT_ORDER:
            record: dict[str, Any] = {"group": group, "expert": expert}
            passes = True
            normalized_gains = []
            for source in sources:
                frame = frames[source]
                target = frame[TARGET].to_numpy(np.float64)
                months = frame["game_month"].to_numpy(np.int16)
                regular = frame["game_type"].astype(str).eq("R").to_numpy()
                selected = regular & (keys[source] == group)
                n_rows = int(selected.sum())
                candidate = np.clip(
                    parents[source] + directions[source][expert], 0.001, 0.999
                )
                gain = _sse_gain(target, parents[source], candidate, selected)
                unique_months = np.unique(months[selected])
                positive_months = sum(
                    _sse_gain(
                        target, parents[source], candidate,
                        selected & (months == month),
                    ) > 0.0
                    for month in unique_months
                )
                required = (len(unique_months) + 1) // 2
                passes = bool(
                    passes
                    and n_rows >= MIN_GROUP_ROWS
                    and gain > 0.0
                    and positive_months >= required
                )
                normalized_gains.append(gain / max(n_rows, 1))
                record.update(
                    {
                        f"{source}_rows": n_rows,
                        f"{source}_sse_gain": gain,
                        f"{source}_positive_months": int(positive_months),
                        f"{source}_required_positive_months": int(required),
                    }
                )
            record["passes"] = passes
            record["minimum_sse_gain_per_row"] = float(min(normalized_gains))
            records.append(record)
            if passes:
                evidence.append(record)
        if evidence:
            selected = max(
                evidence,
                key=lambda item: (
                    item["minimum_sse_gain_per_row"],
                    -EXPERT_ORDER.index(item["expert"]),
                ),
            )
            table[group] = str(selected["expert"])
    return table, pd.DataFrame(records)


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
    archetypes = {name: build_archetypes(frame) for name, frame in frames.items()}
    directions = {
        name: {
            expert: _candidate_minus_parent(path, name)
            for expert, path in expert_paths.items()
        }
        for name in frames
    }

    trials: list[dict[str, Any]] = []
    routers: dict[tuple[str, ...], dict[str, str]] = {}
    diagnostics: list[pd.DataFrame] = []
    for schema in SCHEMAS:
        keys = {name: schema_key(archetypes[name], schema) for name in frames}
        router, evidence = stable_router(frames, parents, directions, keys)
        routers[schema] = router
        evidence.insert(0, "schema", "+".join(schema))
        diagnostics.append(evidence)
        source_metrics: dict[str, Any] = {}
        for source in ("full_2022", "late_2023"):
            regular = frames[source]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[source], directions[source], keys[source], router, regular
            )
            source_metrics[source] = axis_metrics(
                frames[source], parents[source], candidate, active
            )
        gains = [source_metrics[source]["gain"] for source in source_metrics]
        eligible = bool(
            router
            and min(gains) > 0.0
            and source_metrics["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
            and source_metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
        )
        trials.append(
            {
                "schema": "+".join(schema),
                "complexity": len(schema),
                "eligible": eligible,
                "selected_cells": len(router),
                "minimum_source_gain": float(min(gains)),
                "mean_source_gain": float(np.mean(gains)),
                "source_metrics": source_metrics,
            }
        )
    pd.concat(diagnostics, ignore_index=True).to_csv(
        output_dir / "two_origin_cell_evidence.csv", index=False
    )
    eligible_trials = [trial for trial in trials if trial["eligible"]]
    pool = eligible_trials if eligible_trials else trials
    selected = max(
        pool,
        key=lambda trial: (
            trial["eligible"], trial["minimum_source_gain"],
            trial["mean_source_gain"], -trial["complexity"],
        ),
    )
    selected_schema = next(
        schema for schema in SCHEMAS if "+".join(schema) == selected["schema"]
    )
    router = routers[selected_schema]
    pd.DataFrame(
        [{"group": group, "expert": expert} for group, expert in sorted(router.items())]
    ).to_csv(output_dir / "selected_router.csv", index=False)

    candidates: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    assignments: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        keys = schema_key(archetypes[name], selected_schema)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        candidates[name], actives[name], assignments[name] = apply_router(
            parents[name], directions[name], keys, router, regular
        )
        metrics[name] = axis_metrics(
            frame, parents[name], candidates[name], actives[name]
        )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidates["full_2024"],
        actives["full_2024"], [parents["full_2024"], candidates["full_2024"]],
    )
    assignment_counts = {
        name: {
            expert: int(np.sum(assignments[name] == expert))
            for expert in ("PARENT", *EXPERT_ORDER)
            if np.any(assignments[name] == expert)
        }
        for name in frames
    }
    source_pass = bool(selected["eligible"])
    locked_pass = bool(
        source_pass
        and metrics["full_2024"]["gain"] >= 3.0
        and metrics["full_2024"]["positive_month_fraction"] >= 0.625
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"active_{name}": actives[name] for name in frames},
        **{f"assignment_{name}": assignments[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "selected_schema": selected,
        "schema_trials": trials,
        "selected_router": router,
        "assignment_counts": assignment_counts,
        "metrics": metrics,
        "locked_robustness": robustness,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "organizer_data_only": True,
            "f_route_protected_as_v320": True,
            "each_cell_expert_positive_in_both_source_origins": True,
            "each_cell_expert_positive_in_majority_source_months": True,
            "schema_selection_source_only": True,
            "full_2024_evaluated_after_router_freeze": True,
            "sequential_followup_after_v328": True,
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
