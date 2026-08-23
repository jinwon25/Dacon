"""Low-rank pitcher-by-count/hand residual screen above frozen v27.

The candidate is intentionally narrow.  For each completed source season it
centres the residual of the train-only wave-0 OOF model, builds a
``pitcher_id x (balls, strikes, batter_hand)`` matrix, applies empirical-Bayes
shrinkage, and reconstructs the matrix with a deterministic truncated SVD.
An audit row receives the equal-weight mean of the independently fitted prior
season effects; an unseen pitcher contributes zero for that source season.

The same smoothing/rank/domain/weight recipe must pass 2022 and late-2023
before full-2024 and late-2024 are inspected.  No evaluation-row aggregate,
test row, current-fold label, or external league-rate estimate enters a
feature or prediction.
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
from src.core.banks import _metadata


TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
SMOOTHING_GRID = (300.0, 600.0)
RANK_GRID = (2, 4, 6)
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
CONSENSUS_KEYS = ("signal", "domain", "weight")


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    balls = pd.to_numeric(output["balls_before"], errors="raise").to_numpy(
        np.int16
    )
    strikes = pd.to_numeric(
        output["strikes_before"], errors="raise"
    ).to_numpy(np.int16)
    hand = pd.to_numeric(output["batter_hand"], errors="raise").to_numpy(
        np.int16
    )
    if not (
        np.isin(balls, np.arange(4)).all()
        and np.isin(strikes, np.arange(3)).all()
        and np.isin(hand, (1, 2)).all()
    ):
        raise ValueError("unexpected count or batter-hand value")
    output["context_position"] = (
        (balls * 3 + strikes) * 2 + (hand - 1)
    ).astype(np.int8)
    return output


def eligible_source_years(audit_year: int) -> tuple[int, ...]:
    """Return only completed OOF seasons strictly before an audit origin."""

    return tuple(year for year in SOURCE_YEARS if year < int(audit_year))


def fit_source_matrix(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    *,
    smoothing_grid: tuple[float, ...] = SMOOTHING_GRID,
    rank_grid: tuple[int, ...] = RANK_GRID,
) -> dict[str, object]:
    """Fit centred EB matrices and their deterministic SVD reconstructions."""

    source = _prepare(rows).reset_index(drop=True)
    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    if not (len(source) == len(target) == len(parent)):
        raise ValueError("source row/prediction length mismatch")
    if source["pitcher_id"].isna().any():
        raise ValueError("missing source pitcher_id")
    residual = target - parent
    residual_mean = float(residual.mean())
    residual = residual - residual_mean

    codes, pitcher_ids = pd.factorize(source["pitcher_id"], sort=True)
    contexts = source["context_position"].to_numpy(np.int16)
    shape = (len(pitcher_ids), 24)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(sums, (codes, contexts), residual)
    np.add.at(counts, (codes, contexts), 1)

    reconstructions: dict[tuple[float, int], np.ndarray] = {}
    retained_energy: dict[str, dict[str, float]] = {}
    for smoothing in smoothing_grid:
        saturated = sums / (counts.astype(np.float64) + float(smoothing))
        left, singular, right = np.linalg.svd(saturated, full_matrices=False)
        total = float(np.square(singular).sum())
        retained_energy[str(int(smoothing))] = {}
        for rank in rank_grid:
            effective_rank = min(int(rank), len(singular))
            reconstruction = (
                left[:, :effective_rank] * singular[:effective_rank]
            ) @ right[:effective_rank, :]
            reconstructions[(float(smoothing), int(rank))] = reconstruction
            retained = float(np.square(singular[:effective_rank]).sum())
            retained_energy[str(int(smoothing))][str(int(rank))] = (
                retained / total if total > 0.0 else 0.0
            )

    return {
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "reconstructions": reconstructions,
        "diagnostics": {
            "rows": int(len(source)),
            "pitchers": int(len(pitcher_ids)),
            "observed_cells": int((counts > 0).sum()),
            "cell_density": float((counts > 0).mean()),
            "residual_mean_before_centering": residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "retained_energy": retained_energy,
        },
    }


def map_source_matrix(
    model: dict[str, object], rows: pd.DataFrame
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Map one source-season matrix; unseen pitchers receive exactly zero."""

    query = _prepare(rows).reset_index(drop=True)
    pitcher_ids = np.asarray(model["pitcher_ids"])
    source_index = pd.Index(pitcher_ids).get_indexer(query["pitcher_id"])
    seen = source_index >= 0
    safe_index = np.where(seen, source_index, 0)
    contexts = query["context_position"].to_numpy(np.int16)
    counts = np.asarray(model["counts"])
    exact_seen = seen & (counts[safe_index, contexts] > 0)
    output: dict[str, np.ndarray] = {}
    for (smoothing, rank), matrix in dict(model["reconstructions"]).items():
        values = np.zeros(len(query), dtype=np.float64)
        values[seen] = np.asarray(matrix)[
            source_index[seen], contexts[seen]
        ]
        output[f"lowrank_s{int(smoothing)}_r{int(rank)}"] = values
    return output, seen, exact_seen


def _load_rows(project: Path) -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "game_month",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
        TARGET,
    ]
    raw = pd.read_csv(
        project / "data" / "train.csv",
        usecols=columns,
        low_memory=False,
    )
    raw = _add_domain_and_pressure(raw)
    return {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in range(2020, 2025)
    }


def _load_source_oof(
    project: Path, rows: dict[int, pd.DataFrame]
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    models: dict[int, dict[str, object]] = {}
    diagnostics_by_year: dict[str, object] = {}
    for year in SOURCE_YEARS:
        path = (
            project
            / "artifacts"
            / "followup"
            / "oof"
            / f"wave0_incumbent_validate_{year}.npz"
        )
        with np.load(path, allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        expected = rows[year][TARGET].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"wave0 source target/order mismatch for {year}")
        model = fit_source_matrix(rows[year], target, parent)
        models[year] = model
        diagnostics_by_year[str(year)] = model["diagnostics"]
    return models, diagnostics_by_year


def build_audit_bank(
    models: dict[int, dict[str, object]],
    rows: pd.DataFrame,
    audit_year: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Average independent prior-season effects, including unseen-source zero."""

    source_years = eligible_source_years(audit_year)
    if not source_years:
        raise ValueError(f"no prior source OOF for audit {audit_year}")
    mapped = {
        year: map_source_matrix(models[year], rows) for year in source_years
    }
    names = sorted(mapped[source_years[0]][0])
    bank = {
        name: np.mean(
            np.vstack([mapped[year][0][name] for year in source_years]),
            axis=0,
        )
        for name in names
    }
    seen_matrix = np.vstack([mapped[year][1] for year in source_years])
    exact_matrix = np.vstack([mapped[year][2] for year in source_years])
    coverage = {
        "source_years": list(source_years),
        "rows": int(len(rows)),
        "pitcher_seen_any_rate": float(seen_matrix.any(axis=0).mean()),
        "pitcher_seen_every_rate": float(seen_matrix.all(axis=0).mean()),
        "exact_context_seen_any_rate": float(exact_matrix.any(axis=0).mean()),
        "exact_context_seen_every_rate": float(exact_matrix.all(axis=0).mean()),
        "per_source_pitcher_seen_rate": {
            str(year): float(mapped[year][1].mean()) for year in source_years
        },
    }
    return bank, coverage


def _frame(
    target: np.ndarray, month: np.ndarray, domain: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    signal: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    candidate = np.asarray(parent, dtype=np.float64).copy()
    candidate[active] = np.clip(
        candidate[active]
        + float(weight) * np.asarray(signal, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return candidate, active


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, bank: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, signal in bank.items():
        for route in ROUTES:
            for weight in WEIGHTS:
                candidate, active = _candidate(
                    frame, parent, signal, route, weight
                )
                result = diagnostics(frame, parent, candidate, active)
                applied_domain = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                rows.append(
                    {
                        "signal": name,
                        "domain": route,
                        "weight": float(weight),
                        "gain": float(result["gain"]),
                        "positive_month_fraction": float(
                            result["positive_month_fraction"]
                        ),
                        "worst_month_gain": float(
                            result["worst_month_gain"]
                        ),
                        "minimum_domain_gain": float(
                            result["minimum_domain_gain"]
                        ),
                        "applied_domain_gain": float(applied_domain),
                        "selection_score": float(
                            min(
                                result["gain"],
                                result["worst_month_gain"],
                                applied_domain,
                            )
                        ),
                        "mean_abs_shift": float(result["mean_abs_shift"]),
                    }
                )
    return pd.DataFrame(rows)


def select_consensus(
    stage1: pd.DataFrame, stage2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Select one exact recipe from 2022 and late-2023 only."""

    merged = stage1.merge(
        stage2, on=list(CONSENSUS_KEYS), suffixes=("_2022", "_2023")
    )
    if merged.empty:
        raise ValueError("no exact recipe shared by selection origins")
    merged["minimum_gain"] = merged[["gain_2022", "gain_2023"]].min(axis=1)
    merged["minimum_worst_month"] = merged[
        ["worst_month_gain_2022", "worst_month_gain_2023"]
    ].min(axis=1)
    merged["minimum_applied_domain"] = merged[
        ["applied_domain_gain_2022", "applied_domain_gain_2023"]
    ].min(axis=1)
    merged["consensus_score"] = merged[
        ["minimum_gain", "minimum_worst_month", "minimum_applied_domain"]
    ].min(axis=1)
    merged["passes_consensus_gate"] = (
        merged["gain_2022"].gt(0.0)
        & merged["gain_2023"].gt(0.0)
        & merged["positive_month_fraction_2022"].ge(0.75)
        & merged["positive_month_fraction_2023"].eq(1.0)
        & merged["minimum_worst_month"].gt(0.0)
        & merged["minimum_applied_domain"].gt(0.0)
    )
    merged = merged.sort_values(
        ["passes_consensus_gate", "consensus_score", "minimum_gain"],
        ascending=False,
    ).reset_index(drop=True)
    passing = merged.loc[merged["passes_consensus_gate"]]
    chosen = passing.iloc[0] if len(passing) else merged.iloc[0]
    return merged, chosen


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(project)
    models, source_diagnostics = _load_source_oof(project, rows)
    banks: dict[int, dict[str, np.ndarray]] = {}
    coverage: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        banks[year], coverage[str(year)] = build_audit_bank(
            models, rows[year], year
        )

    meta22 = _metadata(project, 2022)
    expected22 = rows[2022][TARGET].to_numpy(np.float64)
    if not np.array_equal(meta22["target"], expected22):
        raise ValueError("2022 audit target/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    stage1 = _screen(frame22, meta22["parent"], banks[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw
    late23_mask = rows[2023]["game_month"].ge(8).to_numpy()
    frame23 = axes["selection_late_2023"]
    expected23 = rows[2023].loc[late23_mask, TARGET].to_numpy(np.float64)
    if not np.array_equal(frame23["target"].to_numpy(np.float64), expected23):
        raise ValueError("late-2023 audit target/order mismatch")
    bank23 = {name: value[late23_mask] for name, value in banks[2023].items()}
    stage2 = _screen(frame23, v27_parent(frame23), bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    late24_mask = rows[2024]["game_month"].ge(8).to_numpy()
    audits: dict[str, dict[str, object]] = {}
    for axis_name, signal in (
        ("outer_full_2024", banks[2024][recipe["signal"]]),
        (
            "replication_late_2024",
            banks[2024][recipe["signal"]][late24_mask],
        ),
    ):
        frame = axes[axis_name]
        parent = v27_parent(frame)
        candidate, active = _candidate(
            frame, parent, signal, recipe["domain"], recipe["weight"]
        )
        audits[axis_name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            signal=signal,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": audits["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": audits["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": audits["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": audits["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": audits[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
    }
    summary = {
        "protocol": "V50_TWO_ORIGIN_LOW_RANK_PITCHER_CONTEXT_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "candidate_origin": (
            "independent public-repository pattern, reimplemented using only "
            "local train-only wave0 OOF residuals"
        ),
        "source_years": list(SOURCE_YEARS),
        "source_model": (
            "season-centred wave0 OOF residual; pitcher x 24 count/hand EB "
            "matrix; deterministic truncated SVD"
        ),
        "grid": {
            "smoothing": list(SMOOTHING_GRID),
            "rank": list(RANK_GRID),
            "routes": list(ROUTES),
            "weights": list(WEIGHTS),
        },
        "source_diagnostics": source_diagnostics,
        "coverage": coverage,
        "selection": "exact recipe consensus on 2022 and late-2023 only",
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                chosen["worst_month_gain_2023"]
            ),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": audits,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_rate_used": False,
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
        default=Path("artifacts/v50_low_rank_pitcher_context_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
