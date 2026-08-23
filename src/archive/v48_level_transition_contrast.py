"""Season-forward league-level transition residual contrasts above v27.

The Futures/Regular regime mean is highly non-stationary, so absolute group
rates cannot be transferred safely.  This experiment instead estimates only
the residual contrast of a player's previous-season level transition relative
to the source model's mean residual in the same deployment domain.

Three successive, non-overlapping transfers are used:

* late 2021 residual contrast -> late 2022 selection;
* late 2022 residual contrast -> late 2023 selection;
* late 2023 residual contrast -> full/late 2024 audit.

The exact same signal, domain route, and blend weight must pass both selection
origins before 2024 is opened.  Previous-season R/F shares are target-free,
and audit labels or other audit rows are never used to construct a signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import read_main
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata, grid_rows
from src.archive.v36_two_origin_consensus import select_consensus


TARGET = "control_success"
GROUP_ALPHA = 1000.0
SIGNALS = (
    "pitcher_transition",
    "batter_transition",
    "crossed_transition",
)


def previous_level(
    train: pd.DataFrame, year: int, entity: str
) -> pd.Series:
    """Map entities to R/F/mixed using only the immediately prior season."""

    history = train.loc[train["season"].eq(year - 1), [entity, "game_type"]]
    counts = history.groupby([entity, "game_type"], observed=True).size().unstack(
        fill_value=0
    )
    for level in ("R", "F"):
        if level not in counts:
            counts[level] = 0
    total = (counts["R"] + counts["F"]).clip(lower=1)
    regular_share = counts["R"] / total
    return pd.Series(
        np.select(
            [regular_share.ge(0.80), regular_share.le(0.20)],
            ["R", "F"],
            default="MIX",
        ),
        index=counts.index,
        dtype="string",
    )


def transition_key(
    train: pd.DataFrame, rows: pd.DataFrame, year: int, entity: str
) -> pd.Series:
    """Return prior-level -> current-level without consulting current labels."""

    prior = rows[entity].map(previous_level(train, year, entity)).fillna("NEW")
    current = rows["game_type"].astype("string").fillna("__MISSING__")
    return (prior.astype(str) + ">" + current.astype(str)).astype("string")


def residual_contrast(
    train: pd.DataFrame,
    source_rows: pd.DataFrame,
    source_target: np.ndarray,
    source_parent: np.ndarray,
    query_rows: pd.DataFrame,
    *,
    source_year: int,
    query_year: int,
    entity: str,
    alpha: float = GROUP_ALPHA,
) -> np.ndarray:
    """Estimate EB group residual minus its source-domain residual mean."""

    if not (
        len(source_rows) == len(source_target) == len(source_parent)
    ):
        raise ValueError("source residual arrays differ in length")
    source = pd.DataFrame(
        {
            "transition": transition_key(
                train, source_rows, source_year, entity
            ).to_numpy(),
            "domain3": source_rows["domain3"].astype(str).to_numpy(),
            "residual": np.asarray(source_target, dtype=np.float64)
            - np.asarray(source_parent, dtype=np.float64),
        }
    )
    domain = source.groupby("domain3", observed=True)["residual"].mean()
    stats = source.groupby(
        ["transition", "domain3"], observed=True
    )["residual"].agg(n="size", s="sum")
    domain_prior = stats.index.get_level_values("domain3").map(domain).to_numpy(
        np.float64
    )
    stats["contrast"] = (
        (stats["s"].to_numpy(np.float64) + float(alpha) * domain_prior)
        / (stats["n"].to_numpy(np.float64) + float(alpha))
        - domain_prior
    )
    query_index = pd.MultiIndex.from_arrays(
        [
            transition_key(train, query_rows, query_year, entity).to_numpy(),
            query_rows["domain3"].astype(str).to_numpy(),
        ],
        names=("transition", "domain3"),
    )
    return stats["contrast"].reindex(query_index).fillna(0.0).to_numpy(np.float64)


def signal_bank(
    train: pd.DataFrame,
    source_rows: pd.DataFrame,
    source_target: np.ndarray,
    source_parent: np.ndarray,
    query_rows: pd.DataFrame,
    *,
    source_year: int,
    query_year: int,
) -> dict[str, np.ndarray]:
    pitcher = residual_contrast(
        train,
        source_rows,
        source_target,
        source_parent,
        query_rows,
        source_year=source_year,
        query_year=query_year,
        entity="pitcher_id",
    )
    batter = residual_contrast(
        train,
        source_rows,
        source_target,
        source_parent,
        query_rows,
        source_year=source_year,
        query_year=query_year,
        entity="batter_id",
    )
    return {
        "pitcher_transition": pitcher,
        "batter_transition": batter,
        "crossed_transition": 0.75 * pitcher + 0.25 * batter,
    }


def _selection_frame(
    target: np.ndarray, month: np.ndarray, domain: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _screen(
    frame: pd.DataFrame,
    parent: np.ndarray,
    bank: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, correction in bank.items():
        raw = np.clip(parent + correction, 0.001, 0.999)
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
    frame: pd.DataFrame,
    correction: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + float(weight) * correction[active], 0.001, 0.999
    )
    return output, active


def _wave0_2021(
    project: Path, train: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / "wave0_incumbent_validate_2021.npz"
    )
    with np.load(path) as saved:
        index = saved["valid_idx"].astype(np.int64)
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    rows = train.iloc[index].reset_index(drop=True)
    if not np.array_equal(rows[TARGET].to_numpy(np.float64), target):
        raise ValueError("wave0 2021 target alignment failure")
    return rows, target, parent


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        read_main(project / "data" / "train.csv")
    )

    # Selection origin 1: late-2021 residuals -> late-2022.
    rows21, target21, parent21 = _wave0_2021(project, train)
    late21 = rows21["game_month"].ge(8).to_numpy()
    meta22 = _metadata(project, 2022)
    rows22 = train.loc[train["season"].eq(2022)].reset_index(drop=True)
    late22 = rows22["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        rows22[TARGET].to_numpy(np.float64), meta22["target"]
    ):
        raise ValueError("2022 metadata target alignment failure")
    query22 = rows22.loc[late22].reset_index(drop=True)
    bank22 = signal_bank(
        train,
        rows21.loc[late21].reset_index(drop=True),
        target21[late21],
        parent21[late21],
        query22,
        source_year=2021,
        query_year=2022,
    )
    frame22 = _selection_frame(
        meta22["target"][late22],
        meta22["month"][late22],
        meta22["domain"][late22],
    )
    stage1 = _screen(frame22, meta22["parent"][late22], bank22)
    stage1.to_csv(output_dir / "selection_late_2022.csv", index=False)

    # Selection origin 2: late-2022 residuals -> late-2023.
    axes = _cached_v25_axes(project, train)
    rows23 = train.loc[train["season"].eq(2023)].reset_index(drop=True)
    late23 = rows23["game_month"].ge(8).to_numpy()
    query23 = rows23.loc[late23].reset_index(drop=True)
    bank23 = signal_bank(
        train,
        query22,
        meta22["target"][late22],
        meta22["parent"][late22],
        query23,
        source_year=2022,
        query_year=2023,
    )
    frame23 = axes["selection_late_2023"]
    if not np.array_equal(
        query23[TARGET].to_numpy(np.float64),
        frame23["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target alignment failure")
    stage2 = _screen(frame23, v27_parent(frame23), bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    # Frozen audit: late-2023 residuals -> 2024.
    rows24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    bank24 = signal_bank(
        train,
        query23,
        frame23["target"].to_numpy(np.float64),
        v27_parent(frame23),
        rows24,
        source_year=2023,
        query_year=2024,
    )
    correction24 = bank24[recipe["signal"]]
    late24 = rows24["game_month"].ge(8).to_numpy()
    results: dict[str, dict[str, object]] = {}
    for axis, correction in (
        ("outer_full_2024", correction24),
        ("replication_late_2024", correction24[late24]),
    ):
        frame = axes[axis]
        if len(frame) != len(correction):
            raise ValueError(f"audit correction row mismatch: {axis}")
        candidate, active = _candidate(
            frame, correction, recipe["domain"], recipe["weight"]
        )
        results[axis] = diagnostics(
            frame, v27_parent(frame), candidate, active
        )
        np.savez_compressed(
            output_dir / f"{axis}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            correction=correction,
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
        "protocol": "V48_THREE_ORIGIN_LEVEL_TRANSITION_CONTRAST_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "signal_contract": (
            "previous-season R/F share is target-free; source OOF residual "
            "group mean is centered by source-domain residual mean"
        ),
        "group_alpha": GROUP_ALPHA,
        "chosen": {
            **recipe,
            "gain_late_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_late_2022": float(chosen["worst_month_gain_2022"]),
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
        default=Path("artifacts/v48_level_transition_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
