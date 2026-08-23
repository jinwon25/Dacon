"""Four-origin reliability gate for the v39 hierarchical season forecast.

2021 is used only to prefilter raw formula identities against a simple ASOF
prior.  The exact formula, minimum current-season pitcher sample, deployment
domain, and blend weight must then agree on 2022 and late-2023.  The recipe is
frozen before it is evaluated on 2024.

This is a follow-up from the v39 family, whose 2024 result has already been
seen.  Consequently even a numerical pass is labelled family-reuse evidence,
not a pristine promotion audit.
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
from src.v35_three_stage_multibank import DOMAINS, _candidate
from src.v36_two_origin_consensus import select_consensus
from src.v39_hierarchical_season_forecast import (
    TARGET,
    current_season_counts,
    forecast_bank,
)


MINIMUM_PITCHER_SEASON_N = (0.0, 25.0, 75.0, 150.0, 300.0)
PREFILTER_QUOTA = 2


def formula_family(name: str) -> str:
    body = str(name).split("::", 1)[1]
    scope, kind, *_ = body.split("_")
    return f"{scope}_{kind}"


def asof_prior(rows: pd.DataFrame) -> np.ndarray:
    pitcher = (
        pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    batter = (
        pd.to_numeric(rows["asof_batter_success_rate"], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    return np.clip(0.75 * pitcher + 0.25 * batter, 0.001, 0.999)


def prefilter_formulas(
    frame: pd.DataFrame,
    bank: dict[str, np.ndarray],
    *,
    quota: int = PREFILTER_QUOTA,
) -> tuple[pd.DataFrame, list[str]]:
    parent = asof_prior(frame)
    active = np.ones(len(frame), dtype=bool)
    rows = []
    for name, candidate in bank.items():
        result = diagnostics(frame, parent, candidate, active)
        score = min(
            float(result["gain"]),
            float(result["worst_month_gain"]),
            float(result["minimum_domain_gain"]),
        )
        rows.append(
            {
                "signal": name,
                "family": formula_family(name),
                "selection_score": score,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["selection_score", "gain"], ascending=False
    )
    selected: list[str] = []
    for family in sorted(metrics["family"].unique()):
        selected.extend(
            metrics.loc[metrics["family"].eq(family)]
            .head(quota)["signal"]
            .astype(str)
            .tolist()
        )
    return metrics, selected


def reliability_raw(
    raw: np.ndarray,
    parent: np.ndarray,
    season_n: np.ndarray,
    minimum_n: float,
) -> np.ndarray:
    if not (len(raw) == len(parent) == len(season_n)):
        raise ValueError("reliability arrays have different lengths")
    return np.where(np.asarray(season_n) >= minimum_n, raw, parent)


def gated_name(signal: str, minimum_n: float) -> str:
    return f"{signal}__minn{int(minimum_n)}"


def split_gated_name(name: str) -> tuple[str, float]:
    signal, value = str(name).rsplit("__minn", 1)
    return signal, float(value)


def selection_grid(
    frame: pd.DataFrame,
    parent: np.ndarray,
    bank: dict[str, np.ndarray],
    season_n: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal, raw in bank.items():
        for minimum_n in MINIMUM_PITCHER_SEASON_N:
            gated = reliability_raw(raw, parent, season_n, minimum_n)
            name = gated_name(signal, minimum_n)
            for route in DOMAINS:
                rows.extend(
                    grid_rows(
                        frame,
                        parent,
                        parent,
                        gated,
                        signal=name,
                        direction_mode="toward_parent",
                        route=route,
                    )
                )
    return pd.DataFrame(rows)


def _season_n(train: pd.DataFrame, year: int) -> np.ndarray:
    history = train.loc[train["season"].lt(year)].reset_index(drop=True)
    query = train.loc[train["season"].eq(year)].reset_index(drop=True)
    return current_season_counts(history, query, "pitcher_id")[0]


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )

    rows21 = train.loc[train["season"].eq(2021)].reset_index(drop=True)
    frame21 = rows21.copy()
    frame21["target"] = frame21[TARGET].to_numpy(np.float64)
    bank21 = forecast_bank(train, 2021)
    prefilter_metrics, selected_formulas = prefilter_formulas(frame21, bank21)
    prefilter_metrics.to_csv(output_dir / "prefilter_2021.csv", index=False)

    banks = {
        year: {
            name: value
            for name, value in forecast_bank(train, year).items()
            if name in selected_formulas
        }
        for year in (2022, 2023, 2024)
    }
    n_by_year = {year: _season_n(train, year) for year in (2022, 2023, 2024)}

    meta22 = _metadata(project, 2022)
    frame22 = pd.DataFrame(
        {
            "target": meta22["target"],
            "game_month": meta22["month"],
            "domain3": meta22["domain"],
        }
    )
    stage1 = selection_grid(frame22, meta22["parent"], banks[2022], n_by_year[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    axes = _cached_v25_axes(project, train)
    selection23 = axes["selection_late_2023"]
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    bank23 = {name: value[late23] for name, value in banks[2023].items()}
    stage2 = selection_grid(
        selection23,
        v27_parent(selection23),
        bank23,
        n_by_year[2023][late23],
    )
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, selected = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)

    selected_signal, selected_minimum_n = split_gated_name(selected["signal"])
    recipe = {
        "signal": str(selected["signal"]),
        "direction": "toward_parent",
        "domain": str(selected["domain"]),
        "weight": float(selected["weight"]),
    }
    full24_month = train.loc[train["season"].eq(2024), "game_month"].to_numpy()
    late24 = full24_month >= 8
    results: dict[str, dict[str, object]] = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        parent = v27_parent(frame)
        raw = banks[2024][selected_signal]
        season_n = n_by_year[2024]
        if axis_name == "replication_late_2024":
            raw = raw[late24]
            season_n = season_n[late24]
        gated = reliability_raw(raw, parent, season_n, selected_minimum_n)
        candidate, active = _candidate(
            frame, {recipe["signal"]: gated}, [recipe]
        )
        results[axis_name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            raw=raw,
            season_n=season_n,
            gated=gated,
            candidate=candidate,
            active=active,
            game_month=frame["game_month"].to_numpy(np.int16),
            domain3=frame["domain3"].astype(str).to_numpy(),
        )

    numerical_gates = {
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
        "protocol": "V40_FOUR_ORIGIN_RELIABILITY_GATED_HIERARCHY_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "prefilter_axis": "2021 raw formula vs row-local ASOF prior",
        "selection_axes": "2022 and late-2023 exact recipe consensus",
        "audit_axis": "full-2024; late-2024 secondary",
        "prefiltered_formula_count": len(selected_formulas),
        "prefiltered_formulas": selected_formulas,
        "consensus_recipe_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "chosen": {
            **recipe,
            "base_signal": selected_signal,
            "minimum_pitcher_season_n": selected_minimum_n,
            "gain_2022": float(selected["gain_2022"]),
            "gain_late_2023": float(selected["gain_2023"]),
            "worst_month_2022": float(selected["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                selected["worst_month_gain_2023"]
            ),
            "consensus_score": float(selected["consensus_score"]),
        },
        "audits": results,
        "numerical_gates": numerical_gates,
        "numerically_eligible": bool(all(numerical_gates.values())),
        "eligible_for_packaging": False,
        "promotion_block": "2024 labels were previously viewed for the v39 family",
        "family_reused_audit_risk": True,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_recipe_selection": False,
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
        default=Path("artifacts/v40_reliability_hierarchy_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
