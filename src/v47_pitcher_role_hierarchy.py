"""Previous-season pitcher role/context hierarchy above frozen v27.

Pitcher command can differ between starter-like early innings and bullpen-like
late innings.  Existing state models contain pitcher identity and inning as
separate fields but do not expose a low-variance pitcher-by-inning posterior.
This screen builds that posterior from only the two seasons before each audit
origin, with explicit empirical-Bayes backoff:

domain -> pitcher x domain -> pitcher x domain x role context.

The same signal, route, and weight must improve 2022 and late 2023 before the
frozen recipe is audited on full and late 2024.  Deployment is a train-only
lookup mapped independently to each evaluation row.
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
from src.v36_two_origin_consensus import select_consensus


TARGET = "control_success"
SIGNALS: dict[str, tuple[str, ...] | None] = {
    "role_pitcher_domain": None,
    "role_pitcher_inning": ("inning_bucket",),
    "role_pitcher_inning_hand": ("inning_bucket", "batter_hand"),
    "role_pitcher_inning_pressure": ("inning_bucket", "pressure"),
}
PITCHER_ALPHA = 160.0
CONTEXT_ALPHA = {
    "role_pitcher_inning": 70.0,
    "role_pitcher_inning_hand": 55.0,
    "role_pitcher_inning_pressure": 55.0,
}


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string").fillna("__MISSING__")
    for column in ("pitcher_id", "domain3", "batter_hand", "pressure"):
        output[column] = output[column].astype("string").fillna("__MISSING__")
    return output


def _weighted_table(
    history: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    work = history.loc[:, keys].copy()
    work["__w"] = history["__w"].to_numpy(np.float64)
    work["__wy"] = work["__w"] * history[TARGET].to_numpy(np.float64)
    return work.groupby(keys, observed=True, sort=False).agg(
        n=("__w", "sum"), s=("__wy", "sum")
    )


def _map(table: pd.DataFrame, query: pd.DataFrame, keys: list[str], column: str) -> np.ndarray:
    index = pd.MultiIndex.from_frame(query[keys])
    return table[column].reindex(index).fillna(0.0).to_numpy(np.float64)


def role_bank(train: pd.DataFrame, audit_year: int) -> dict[str, np.ndarray]:
    """Create four fixed role posteriors from the two previous seasons."""

    history = _prepare(
        train.loc[
            train["season"].between(audit_year - 2, audit_year - 1)
        ].reset_index(drop=True)
    )
    query = _prepare(
        train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
    )
    if history.empty or query.empty:
        raise ValueError(f"missing history/query for origin {audit_year}")
    history["__w"] = np.where(
        history["season"].eq(audit_year - 1).to_numpy(), 1.0, 0.60
    )
    latest = history.loc[history["season"].eq(audit_year - 1)]
    global_rate = float(latest[TARGET].mean())
    domain_rate = latest.groupby("domain3", observed=True)[TARGET].mean()
    domain_prior = (
        query["domain3"].map(domain_rate).fillna(global_rate).to_numpy(np.float64)
    )

    pitcher_keys = ["pitcher_id", "domain3"]
    pitcher_table = _weighted_table(history, pitcher_keys)
    pitcher_n = _map(pitcher_table, query, pitcher_keys, "n")
    pitcher_s = _map(pitcher_table, query, pitcher_keys, "s")
    pitcher = (pitcher_s + PITCHER_ALPHA * domain_prior) / (
        pitcher_n + PITCHER_ALPHA
    )
    output = {"role_pitcher_domain": np.clip(pitcher, 0.001, 0.999)}

    for name, context in SIGNALS.items():
        if context is None:
            continue
        keys = [*pitcher_keys, *context]
        table = _weighted_table(history, keys)
        n = _map(table, query, keys, "n")
        s = _map(table, query, keys, "s")
        alpha = CONTEXT_ALPHA[name]
        output[name] = np.clip(
            (s + alpha * pitcher) / (n + alpha), 0.001, 0.999
        )
    return output


def _frame(target: np.ndarray, month: np.ndarray, domain: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, bank: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for name, raw in bank.items():
        for route in ("ALL", "R_CORE", "R_ANCHOR", "F"):
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


def _candidate(
    frame: pd.DataFrame, raw: np.ndarray, route: str, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    output = parent.copy()
    output[active] = np.clip(
        parent[active]
        + float(weight) * (np.asarray(raw)[active] - parent[active]),
        0.001,
        0.999,
    )
    return output, active


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    banks = {year: role_bank(train, year) for year in (2022, 2023, 2024)}
    meta22 = _metadata(project, 2022)
    expected22 = train.loc[train["season"].eq(2022), TARGET].to_numpy(np.float64)
    if not np.array_equal(meta22["target"], expected22):
        raise ValueError("2022 target order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    stage1 = _screen(frame22, meta22["parent"], banks[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    axes = _cached_v25_axes(project, train)
    frame23 = axes["selection_late_2023"]
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    expected23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8), TARGET
    ].to_numpy(np.float64)
    if not np.array_equal(frame23["target"].to_numpy(np.float64), expected23):
        raise ValueError("late-2023 target order mismatch")
    bank23 = {name: value[late23] for name, value in banks[2023].items()}
    stage2 = _screen(frame23, v27_parent(frame23), bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    late24 = train.loc[train["season"].eq(2024), "game_month"].ge(8).to_numpy()
    results = {}
    for axis, raw in (
        ("outer_full_2024", banks[2024][recipe["signal"]]),
        ("replication_late_2024", banks[2024][recipe["signal"]][late24]),
    ):
        frame = axes[axis]
        candidate, active = _candidate(
            frame, raw, recipe["domain"], recipe["weight"]
        )
        results[axis] = diagnostics(
            frame, v27_parent(frame), candidate, active
        )
        np.savez_compressed(
            output_dir / f"{axis}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            raw=raw,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
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
    }
    summary = {
        "protocol": "V47_TWO_ORIGIN_PITCHER_ROLE_HIERARCHY_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "signals": list(SIGNALS),
        "history": "two immediately preceding seasons; older weight 0.60",
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(chosen["worst_month_gain_2023"]),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": results,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
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
        default=Path("artifacts/v47_pitcher_role_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
