"""Cross-origin baseball-archetype mixture of frozen experts above v320.

The candidate library contains structurally different, already-frozen
directions.  A router is learned independently on full-2022 and late-2023.
For a baseball archetype to receive an expert in the final router, both source
origins must select the same expert for that archetype.  Full-2024 is opened
only after the routing schema and consensus table are frozen.

The Futures route is deliberately protected because v320 already contains a
dedicated recent-F direct model and a low-rank complement.  All routing
features are current-row fields and fixed baseball buckets; evaluation row
order, identifiers and other evaluation rows are never used.
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


PROTOCOL = "V328_BASEBALL_ARCHETYPE_CONSENSUS_MOE_V1"
TARGET = "control_success"
MIN_GROUP_ROWS = 750
EXPERT_ORDER = (
    "season_innovation",
    "completed_anchor",
    "platoon",
    "trackman_physical",
    "lowrank_interaction",
    "beta_binomial",
    "player_transition",
    "joint_role_h1",
)
SCHEMAS = (
    ("route",),
    ("route", "support"),
    ("route", "failure"),
    ("route", "pitchmix"),
    ("route", "count"),
    ("route", "form"),
    ("route", "platoon_state"),
    ("route", "leverage"),
    ("route", "support", "failure"),
    ("route", "failure", "count"),
    ("route", "support", "pitchmix"),
)


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean(np.square(target - prediction)) / (rate * (1.0 - rate)))
    )


def build_archetypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Build fixed, row-local baseball archetypes without fitted cut points."""

    output = pd.DataFrame(index=frame.index)
    regular = frame["game_type"].astype(str).eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    output["route"] = np.select(
        [~regular, anchor], ["F", "R_ANCHOR"], default="R_CORE"
    )

    support = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0)
    output["support"] = pd.cut(
        support,
        [-np.inf, 100.0, 800.0, 3000.0, np.inf],
        labels=["ROOKIE", "DEVELOPING", "ESTABLISHED", "VETERAN"],
        right=False,
    ).astype(str)

    failure_columns = (
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
    )
    failure_labels = np.asarray(["REVERSE", "MIDDLE", "BALL", "STRIKE"])
    failure = frame.loc[:, failure_columns].apply(pd.to_numeric, errors="coerce")
    failure_values = failure.fillna(-np.inf).to_numpy(np.float64)
    output["failure"] = failure_labels[np.argmax(failure_values, axis=1)]
    output.loc[~np.isfinite(failure_values).any(axis=1), "failure"] = "UNKNOWN"

    mix_n = pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0)
    mix_columns = (
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    mix_labels = np.asarray(["FASTBALL", "BREAKING", "OFFSPEED"])
    mix = frame.loc[:, mix_columns].apply(pd.to_numeric, errors="coerce")
    mix_values = mix.fillna(-np.inf).to_numpy(np.float64)
    mix_max = np.max(mix_values, axis=1)
    pitchmix = mix_labels[np.argmax(mix_values, axis=1)].astype(object)
    pitchmix[mix_max < 0.50] = "MIXED"
    pitchmix[mix_n.to_numpy(np.float64) < 100.0] = "UNESTABLISHED"
    output["pitchmix"] = pitchmix

    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    output["count"] = np.select(
        [
            balls.eq(3) & strikes.eq(2),
            balls.eq(3),
            strikes.eq(2),
            balls.gt(strikes),
            strikes.gt(balls),
        ],
        ["FULL", "THREE_BALL", "TWO_STRIKE", "BATTER_AHEAD", "PITCHER_AHEAD"],
        default="EVEN",
    )

    recent1 = pd.to_numeric(
        frame["asof_pitcher_prev1_game_success_rate"], errors="coerce"
    )
    recent5 = pd.to_numeric(
        frame["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    )
    form_delta = recent1 - recent5
    output["form"] = np.select(
        [form_delta.gt(0.05), form_delta.lt(-0.05), form_delta.notna()],
        ["UP", "DOWN", "STABLE"],
        default="UNKNOWN",
    )

    pitcher_hand = frame["pitcher_hand"].astype(str)
    batter_hand = frame["batter_hand"].astype(str)
    output["platoon_state"] = np.where(
        pitcher_hand.eq(batter_hand), "SAME", "OPPOSITE"
    )
    output.loc[pitcher_hand.isin(["nan", "None"]) | batter_hand.isin(["nan", "None"]), "platoon_state"] = "UNKNOWN"

    leverage = pd.to_numeric(frame["li"], errors="coerce")
    output["leverage"] = pd.cut(
        leverage,
        [-np.inf, 0.7, 1.5, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return output


def schema_key(archetypes: pd.DataFrame, schema: tuple[str, ...]) -> np.ndarray:
    return archetypes.loc[:, list(schema)].astype(str).agg("|".join, axis=1).to_numpy()


def _sse_gain(
    target: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray
) -> float:
    residual_parent = target[selected] - parent[selected]
    residual_candidate = target[selected] - candidate[selected]
    return float(np.sum(np.square(residual_parent) - np.square(residual_candidate)))


def fit_router(
    frame: pd.DataFrame,
    parent: np.ndarray,
    directions: dict[str, np.ndarray],
    keys: np.ndarray,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Select a within-group expert with a majority of positive months."""

    target = frame[TARGET].to_numpy(np.float64)
    months = frame["game_month"].to_numpy(np.int16)
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    table: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for group in sorted(np.unique(keys[regular])):
        selected = regular & (keys == group)
        n_rows = int(selected.sum())
        if n_rows < MIN_GROUP_ROWS:
            continue
        required_positive = (len(np.unique(months[selected])) + 1) // 2
        best: tuple[float, str, int] | None = None
        for expert in EXPERT_ORDER:
            candidate = np.clip(parent + directions[expert], 0.001, 0.999)
            total_gain = _sse_gain(target, parent, candidate, selected)
            positive_months = 0
            for month in np.unique(months[selected]):
                month_selected = selected & (months == month)
                positive_months += int(
                    _sse_gain(target, parent, candidate, month_selected) > 0.0
                )
            rows.append(
                {
                    "group": group,
                    "rows": n_rows,
                    "expert": expert,
                    "sse_gain": total_gain,
                    "positive_months": positive_months,
                    "required_positive_months": required_positive,
                }
            )
            eligible = total_gain > 0.0 and positive_months >= required_positive
            if eligible and (best is None or total_gain > best[0]):
                best = (total_gain, expert, positive_months)
        if best is not None:
            table[group] = best[1]
    return table, pd.DataFrame(rows)


def apply_router(
    parent: np.ndarray,
    directions: dict[str, np.ndarray],
    keys: np.ndarray,
    table: dict[str, str],
    regular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.asarray(parent, dtype=np.float64).copy()
    assignment = np.full(len(parent), "PARENT", dtype="<U32")
    for group, expert in table.items():
        selected = regular & (keys == group)
        candidate[selected] = np.clip(
            candidate[selected] + directions[expert][selected], 0.001, 0.999
        )
        assignment[selected] = expert
    active = assignment != "PARENT"
    return candidate, active, assignment


def axis_metrics(
    frame: pd.DataFrame, parent: np.ndarray, candidate: np.ndarray, active: np.ndarray
) -> dict[str, Any]:
    axes = {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    if not np.any(active):
        return {
            "gain": 0.0,
            "active_gain": 0.0,
            "active_rows": 0,
            "mean_abs_shift_active": 0.0,
            "rms_shift_active": 0.0,
            "positive_month_fraction": 0.0,
            "worst_month_gain": 0.0,
            "months": [],
        }
    return paired_metrics(axes, parent, candidate, active)


def _candidate_minus_parent(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        candidate = saved[f"candidate_{name}"].astype(np.float64)
        parent_key = f"parent_{name}"
        baseline_key = f"baseline_{name}"
        if parent_key in saved.files:
            parent = saved[parent_key].astype(np.float64)
        elif baseline_key in saved.files:
            parent = saved[baseline_key].astype(np.float64)
        else:
            raise KeyError(f"no parent/baseline key for {name}: {path}")
    return candidate - parent


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
    for name, frame in frames.items():
        if len(frame) != len(parents[name]):
            raise ValueError(f"parent alignment mismatch: {name}")
        if any(len(value) != len(frame) for value in directions[name].values()):
            raise ValueError(f"expert alignment mismatch: {name}")

    schema_results: list[dict[str, Any]] = []
    source_tables: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    source_diagnostics: list[pd.DataFrame] = []
    for schema in SCHEMAS:
        keys = {name: schema_key(archetypes[name], schema) for name in frames}
        tables: dict[str, dict[str, str]] = {}
        for source in ("full_2022", "late_2023"):
            tables[source], diagnostic = fit_router(
                frames[source], parents[source], directions[source], keys[source]
            )
            diagnostic.insert(0, "source", source)
            diagnostic.insert(0, "schema", "+".join(schema))
            source_diagnostics.append(diagnostic)
        source_tables[schema] = tables

        cross_metrics: dict[str, Any] = {}
        for trained, evaluated in (
            ("full_2022", "late_2023"),
            ("late_2023", "full_2022"),
        ):
            regular = frames[evaluated]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[evaluated], directions[evaluated], keys[evaluated],
                tables[trained], regular,
            )
            cross_metrics[f"{trained}_to_{evaluated}"] = axis_metrics(
                frames[evaluated], parents[evaluated], candidate, active
            )

        consensus = {
            group: expert
            for group, expert in tables["full_2022"].items()
            if tables["late_2023"].get(group) == expert
        }
        consensus_metrics: dict[str, Any] = {}
        for source in ("full_2022", "late_2023"):
            regular = frames[source]["game_type"].astype(str).eq("R").to_numpy()
            candidate, active, _ = apply_router(
                parents[source], directions[source], keys[source], consensus, regular
            )
            consensus_metrics[source] = axis_metrics(
                frames[source], parents[source], candidate, active
            )
        cross_gains = [value["gain"] for value in cross_metrics.values()]
        consensus_gains = [value["gain"] for value in consensus_metrics.values()]
        eligible = bool(
            consensus
            and min(cross_gains) > 0.0
            and min(consensus_gains) > 0.0
            and consensus_metrics["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
            and consensus_metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
        )
        schema_results.append(
            {
                "schema": "+".join(schema),
                "complexity": len(schema),
                "eligible": eligible,
                "consensus_groups": len(consensus),
                "minimum_cross_gain": float(min(cross_gains)),
                "minimum_consensus_gain": float(min(consensus_gains)),
                "mean_consensus_gain": float(np.mean(consensus_gains)),
                "cross_metrics": cross_metrics,
                "consensus_metrics": consensus_metrics,
            }
        )

    diagnostic_frame = pd.concat(source_diagnostics, ignore_index=True)
    diagnostic_frame.to_csv(output_dir / "source_expert_diagnostics.csv", index=False)
    eligible = [item for item in schema_results if item["eligible"]]
    pool = eligible if eligible else schema_results
    selected = max(
        pool,
        key=lambda item: (
            item["eligible"], item["minimum_cross_gain"],
            item["minimum_consensus_gain"], -item["complexity"],
        ),
    )
    selected_schema = next(
        schema for schema in SCHEMAS if "+".join(schema) == selected["schema"]
    )
    selected_tables = source_tables[selected_schema]
    consensus = {
        group: expert
        for group, expert in selected_tables["full_2022"].items()
        if selected_tables["late_2023"].get(group) == expert
    }
    pd.DataFrame(
        [{"group": group, "expert": expert} for group, expert in sorted(consensus.items())]
    ).to_csv(output_dir / "selected_consensus_router.csv", index=False)

    candidates: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    assignments: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        keys = schema_key(archetypes[name], selected_schema)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        candidates[name], actives[name], assignments[name] = apply_router(
            parents[name], directions[name], keys, consensus, regular
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
        "schema_results": schema_results,
        "consensus_router": consensus,
        "assignment_counts": assignment_counts,
        "metrics": metrics,
        "locked_robustness": robustness,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "expert_paths": {name: str(path) for name, path in expert_paths.items()},
        "restrictions": {
            "organizer_data_only": True,
            "f_route_protected_as_v320": True,
            "fixed_row_local_baseball_archetypes": True,
            "routers_fitted_independently_on_two_source_origins": True,
            "final_group_requires_same_expert_in_both_origins": True,
            "schema_selected_without_full_2024": True,
            "full_2024_opened_once_after_freeze": True,
            "evaluation_row_order_ids_or_aggregates_used": False,
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
    parser.add_argument("--season-innovation", type=Path, required=True)
    parser.add_argument("--completed-anchor", type=Path, required=True)
    parser.add_argument("--platoon", type=Path, required=True)
    parser.add_argument("--trackman-physical", type=Path, required=True)
    parser.add_argument("--lowrank-interaction", type=Path, required=True)
    parser.add_argument("--beta-binomial", type=Path, required=True)
    parser.add_argument("--player-transition", type=Path, required=True)
    parser.add_argument("--joint-role-h1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expert_paths = {
        "season_innovation": args.season_innovation,
        "completed_anchor": args.completed_anchor,
        "platoon": args.platoon,
        "trackman_physical": args.trackman_physical,
        "lowrank_interaction": args.lowrank_interaction,
        "beta_binomial": args.beta_binomial,
        "player_transition": args.player_transition,
        "joint_role_h1": args.joint_role_h1,
    }
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, expert_paths, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
