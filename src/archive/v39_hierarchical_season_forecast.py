"""Hierarchical current-season forecast screened on two temporal origins.

The official ASOF counters contain exact sufficient statistics immediately
before each row.  For an audit season, labelled rows from earlier seasons are
subtracted from those counters to recover current-season evidence.  That
evidence is shrunk toward player forecasts built from career and latest-season
history.  No other audit row is read.

The same raw model, deployment domain, and blend weight must improve the 2022
and late-2023 selection origins.  Full and late 2024 are opened only after the
recipe is fixed.  The 2022 parent is an older OOF model, so it is used as a
consensus direction check rather than represented as a v27 analogue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata, grid_rows
from src.archive.v35_three_stage_multibank import DOMAINS, _candidate
from src.archive.v36_two_origin_consensus import select_consensus


TARGET = "control_success"
PRIOR_KINDS = ("career", "latest", "blend")
SCOPES = ("global", "domain")
POSTERIOR_STRENGTHS = (40.0, 80.0, 160.0)
PITCHER_WEIGHTS = (0.75, 0.90, 1.00)
ENTITY_ALPHA = {
    "pitcher_id": (160.0, 80.0),
    "batter_id": (240.0, 120.0),
}
ASOF_COLUMNS = {
    "pitcher_id": ("asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter_id": ("asof_batter_n", "asof_batter_success_rate"),
}


def _rounded_success(n: np.ndarray, rate: np.ndarray) -> np.ndarray:
    output = np.zeros(len(n), dtype=np.float64)
    valid = (n > 0.0) & np.isfinite(rate)
    output[valid] = np.rint(n[valid] * rate[valid])
    return output


def current_season_counts(
    history: pd.DataFrame,
    query: pd.DataFrame,
    entity: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover row-local current-season trials and successes exactly."""

    n_column, rate_column = ASOF_COLUMNS[entity]
    historic = history.groupby(entity, observed=True)[TARGET].agg(n="size", s="sum")
    history_n = query[entity].map(historic["n"]).fillna(0.0).to_numpy(np.float64)
    history_s = query[entity].map(historic["s"]).fillna(0.0).to_numpy(np.float64)
    cumulative_n = (
        pd.to_numeric(query[n_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    cumulative_n = np.maximum(cumulative_n, 0.0)
    cumulative_rate = (
        pd.to_numeric(query[rate_column], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    cumulative_s = _rounded_success(cumulative_n, cumulative_rate)
    season_n = np.maximum(cumulative_n - history_n, 0.0)
    season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
    return season_n, season_s


def _mapped_stat(
    table: pd.DataFrame,
    query: pd.DataFrame,
    entity: str,
    scope: str,
    column: str,
) -> np.ndarray:
    if scope == "global":
        return query[entity].map(table[column]).fillna(0.0).to_numpy(np.float64)
    if scope != "domain":
        raise ValueError(f"unknown scope: {scope}")
    index = pd.MultiIndex.from_frame(query[[entity, "domain3"]])
    return table[column].reindex(index).fillna(0.0).to_numpy(np.float64)


def _player_forecasts(
    history: pd.DataFrame,
    query: pd.DataFrame,
    entity: str,
    scope: str,
) -> dict[str, np.ndarray]:
    """Return career/latest forecasts using only seasons before the query."""

    if history.empty:
        raise ValueError("player forecast requires historical rows")
    latest_year = int(history["season"].max())
    latest = history.loc[history["season"].eq(latest_year)]
    group_columns = [entity] if scope == "global" else [entity, "domain3"]
    career_stats = history.groupby(group_columns, observed=True)[TARGET].agg(
        n="size", s="sum"
    )
    latest_stats = latest.groupby(group_columns, observed=True)[TARGET].agg(
        n="size", s="sum"
    )
    latest_global = float(latest[TARGET].mean())
    if scope == "global":
        prior = np.full(len(query), latest_global, dtype=np.float64)
    else:
        by_domain = latest.groupby("domain3", observed=True)[TARGET].mean()
        prior = (
            query["domain3"]
            .map(by_domain)
            .fillna(latest_global)
            .to_numpy(np.float64)
        )
    career_n = _mapped_stat(career_stats, query, entity, scope, "n")
    career_s = _mapped_stat(career_stats, query, entity, scope, "s")
    latest_n = _mapped_stat(latest_stats, query, entity, scope, "n")
    latest_s = _mapped_stat(latest_stats, query, entity, scope, "s")
    career_alpha, latest_alpha = ENTITY_ALPHA[entity]
    career_rate = (career_s + career_alpha * prior) / (
        career_n + career_alpha
    )
    latest_rate = (latest_s + latest_alpha * prior) / (
        latest_n + latest_alpha
    )
    return {
        "career": career_rate,
        "latest": latest_rate,
        "blend": 0.35 * career_rate + 0.65 * latest_rate,
    }


def forecast_bank(train: pd.DataFrame, audit_year: int) -> dict[str, np.ndarray]:
    """Build a fixed, low-degree hierarchy for one audit season."""

    history = train.loc[train["season"].lt(audit_year)].reset_index(drop=True)
    query = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
    if history.empty or query.empty:
        raise ValueError(f"missing history or query for {audit_year}")
    season = {
        entity: current_season_counts(history, query, entity)
        for entity in ASOF_COLUMNS
    }
    forecasts = {
        (entity, scope): _player_forecasts(history, query, entity, scope)
        for entity in ASOF_COLUMNS
        for scope in SCOPES
    }
    output: dict[str, np.ndarray] = {}
    for scope in SCOPES:
        for kind in PRIOR_KINDS:
            pitcher_prior = forecasts[("pitcher_id", scope)][kind]
            batter_prior = forecasts[("batter_id", scope)][kind]
            for strength in POSTERIOR_STRENGTHS:
                pitcher_n, pitcher_s = season["pitcher_id"]
                batter_n, batter_s = season["batter_id"]
                pitcher = (pitcher_s + strength * pitcher_prior) / (
                    pitcher_n + strength
                )
                batter = (batter_s + strength * batter_prior) / (
                    batter_n + strength
                )
                for pitcher_weight in PITCHER_WEIGHTS:
                    name = (
                        f"hier::{scope}_{kind}_k{int(strength)}_"
                        f"p{int(round(100 * pitcher_weight))}"
                    )
                    output[name] = np.clip(
                        pitcher_weight * pitcher
                        + (1.0 - pitcher_weight) * batter,
                        0.001,
                        0.999,
                    )
    return output


def _selection_grid(
    frame: pd.DataFrame,
    parent: np.ndarray,
    bank: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, raw in bank.items():
        for route in DOMAINS:
            rows.extend(
                grid_rows(
                    frame,
                    parent,
                    parent,
                    raw,
                    signal=name,
                    direction_mode="toward_parent",
                    route=route,
                )
            )
    return pd.DataFrame(rows)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    bank = {year: forecast_bank(train, year) for year in (2022, 2023, 2024)}

    meta22 = _metadata(project, 2022)
    selection22 = pd.DataFrame(
        {
            "target": meta22["target"],
            "game_month": meta22["month"],
            "domain3": meta22["domain"],
        }
    )
    if not np.array_equal(
        selection22["target"].to_numpy(np.float64),
        train.loc[train["season"].eq(2022), TARGET].to_numpy(np.float64),
    ):
        raise ValueError("2022 row order mismatch")
    stage1 = _selection_grid(selection22, meta22["parent"], bank[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    axes = _cached_v25_axes(project, train)
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    selection23 = axes["selection_late_2023"]
    if not np.array_equal(
        selection23["target"].to_numpy(np.float64),
        train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8), TARGET
        ].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 row order mismatch")
    late_bank23 = {name: value[late23] for name, value in bank[2023].items()}
    stage2 = _selection_grid(selection23, v27_parent(selection23), late_bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, selected = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(selected["signal"]),
        "direction": "toward_parent",
        "domain": str(selected["domain"]),
        "weight": float(selected["weight"]),
    }

    results: dict[str, dict[str, object]] = {}
    full24_month = train.loc[train["season"].eq(2024), "game_month"].to_numpy()
    late24 = full24_month >= 8
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        local = bank[2024][recipe["signal"]]
        if axis_name == "replication_late_2024":
            local = local[late24]
        candidate, active = _candidate(
            frame, {recipe["signal"]: local}, [recipe]
        )
        parent = v27_parent(frame)
        results[axis_name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            raw=local,
            candidate=candidate,
            active=active,
            game_month=frame["game_month"].to_numpy(np.int16),
            domain3=frame["domain3"].astype(str).to_numpy(),
        )

    gates = {
        "consensus_gate": bool(selected["passes_consensus_gate"]),
        "outer_gain_at_least_5": results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": results["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
        "replication_worst_month_above_minus_10": results[
            "replication_late_2024"
        ]["worst_month_gain"]
        > -10.0,
    }
    summary = {
        "protocol": "V39_HIERARCHICAL_CURRENT_SEASON_FORECAST_TWO_ORIGIN_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "candidate_count": len(bank[2022]),
        "consensus_recipe_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "chosen": {
            **recipe,
            "gain_2022": float(selected["gain_2022"]),
            "gain_late_2023": float(selected["gain_2023"]),
            "worst_month_2022": float(selected["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                selected["worst_month_gain_2023"]
            ),
            "consensus_score": float(selected["consensus_score"]),
        },
        "audits": results,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
        "parent_selection_leakage_claimed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v39_hierarchical_season_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
