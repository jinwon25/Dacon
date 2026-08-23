"""Audit a low-dimensional pitcher-by-balls-ahead interaction above 1158.

The public competition audit found the same ``pitcher x (balls > strikes)``
pattern in more than one independent solution.  This module does not import
their fitted values.  It reconstructs a pure interaction from local,
train-only wave-0 OOF residuals and removes each pitcher's main effect.

The exact recipe is selected on 2022 and late-2023.  Full-2024 and late-2024
are opened only after that recipe is frozen.  The latter three axes use the
exact reconstructed OOF parent of the Public 1158.0745556751 standalone ZIP.
No test-row aggregate, row order, or current-fold label enters inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v50_low_rank_pitcher_context import _frame, _load_rows, select_consensus


TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
SMOOTHING_GRID = (25.0, 75.0, 200.0, 600.0)
HISTORY_MODES = ("all", "recent2", "latest1")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.25, 0.50, 0.75, 1.00, 1.50)
CONSENSUS_KEYS = ("signal", "domain", "weight")


def eligible_source_years(audit_year: int, history_mode: str) -> tuple[int, ...]:
    """Return a predeclared, strictly pre-origin completed-season window."""

    prior = tuple(year for year in SOURCE_YEARS if year < int(audit_year))
    if not prior:
        raise ValueError(f"no completed source season before {audit_year}")
    if history_mode == "all":
        return prior
    if history_mode == "recent2":
        return prior[-2:]
    if history_mode == "latest1":
        return prior[-1:]
    raise ValueError(f"unknown history mode: {history_mode}")


def _balls_ahead(rows: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(rows["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(rows["strikes_before"], errors="raise").to_numpy(
        np.int16
    )
    if not (np.isin(balls, np.arange(4)).all() and np.isin(strikes, np.arange(3)).all()):
        raise ValueError("unexpected balls/strikes state")
    return (balls > strikes).astype(np.int8)


def fit_source_interaction(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    *,
    smoothing_grid: tuple[float, ...] = SMOOTHING_GRID,
) -> dict[str, object]:
    """Fit a shrunk binary contrast with exactly zero pitcher main effect.

    For pitcher ``p``, the raw contrast is the residual mean under
    ``balls > strikes`` minus the residual mean otherwise.  Its effective
    sample size is the harmonic two-sample size ``n0*n1/(n0+n1)``.  The two
    mapped effects are centred by the pitcher's observed state prevalence, so
    their count-weighted average is zero.  A pitcher missing either state has
    no interaction estimate.
    """

    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    if not (len(rows) == len(target) == len(parent)):
        raise ValueError("source row/prediction length mismatch")
    if rows["pitcher_id"].isna().any():
        raise ValueError("missing source pitcher_id")

    state = _balls_ahead(rows)
    codes, pitcher_ids = pd.factorize(rows["pitcher_id"], sort=True)
    residual = target - parent
    residual -= float(residual.mean())
    shape = (len(pitcher_ids), 2)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(sums, (codes, state), residual)
    np.add.at(counts, (codes, state), 1)

    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    total = counts.sum(axis=1).astype(np.float64)
    effective_n = np.divide(
        counts[:, 0].astype(np.float64) * counts[:, 1].astype(np.float64),
        total,
        out=np.zeros(len(total), dtype=np.float64),
        where=total > 0,
    )
    prevalence = np.divide(
        counts[:, 1],
        total,
        out=np.zeros(len(total), dtype=np.float64),
        where=total > 0,
    )
    raw_contrast = means[:, 1] - means[:, 0]

    effects: dict[float, np.ndarray] = {}
    zero_checks: dict[str, float] = {}
    for smoothing in smoothing_grid:
        smoothing = float(smoothing)
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")
        contrast = raw_contrast * effective_n / (effective_n + smoothing)
        local = np.column_stack((-prevalence * contrast, (1.0 - prevalence) * contrast))
        local[effective_n <= 0.0] = 0.0
        effects[smoothing] = local
        weighted = np.sum(local * counts, axis=1)
        zero_checks[str(int(smoothing))] = float(np.max(np.abs(weighted), initial=0.0))

    return {
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "effects": effects,
        "diagnostics": {
            "rows": int(len(rows)),
            "pitchers": int(len(pitcher_ids)),
            "pitchers_with_both_states": int((effective_n > 0.0).sum()),
            "both_state_fraction": float((effective_n > 0.0).mean()),
            "count_weighted_main_effect_max_abs": zero_checks,
        },
    }


def map_source_interaction(
    model: dict[str, object], rows: pd.DataFrame
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Map one completed-season model; unseen or one-state pitchers get zero."""

    state = _balls_ahead(rows)
    pitcher_index = pd.Index(np.asarray(model["pitcher_ids"])).get_indexer(
        rows["pitcher_id"]
    )
    seen = pitcher_index >= 0
    safe_index = np.where(seen, pitcher_index, 0)
    counts = np.asarray(model["counts"])
    both_states = seen & (counts[safe_index, 0] > 0) & (counts[safe_index, 1] > 0)
    output: dict[str, np.ndarray] = {}
    for smoothing, effects in dict(model["effects"]).items():
        values = np.zeros(len(rows), dtype=np.float64)
        values[both_states] = np.asarray(effects)[
            pitcher_index[both_states], state[both_states]
        ]
        output[f"ahead_contrast_s{int(smoothing)}"] = values
    return output, seen, both_states


def _load_source_models(
    project: Path, rows: dict[int, pd.DataFrame]
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    models: dict[int, dict[str, object]] = {}
    audit: dict[str, object] = {}
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
        models[year] = fit_source_interaction(rows[year], target, parent)
        audit[str(year)] = models[year]["diagnostics"]
    return models, audit


def build_audit_bank(
    models: dict[int, dict[str, object]],
    rows: pd.DataFrame,
    audit_year: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build all predeclared history-window signals for one audit origin."""

    mapped = {year: map_source_interaction(model, rows) for year, model in models.items()}
    output: dict[str, np.ndarray] = {}
    coverage: dict[str, object] = {}
    for history_mode in HISTORY_MODES:
        years = eligible_source_years(audit_year, history_mode)
        names = sorted(mapped[years[0]][0])
        for name in names:
            output[f"{history_mode}__{name}"] = np.mean(
                np.vstack([mapped[year][0][name] for year in years]), axis=0
            )
        seen = np.vstack([mapped[year][1] for year in years])
        both = np.vstack([mapped[year][2] for year in years])
        coverage[history_mode] = {
            "source_years": list(years),
            "pitcher_seen_any_rate": float(seen.any(axis=0).mean()),
            "pitcher_seen_every_rate": float(seen.all(axis=0).mean()),
            "both_states_any_rate": float(both.any(axis=0).mean()),
            "both_states_every_rate": float(both.all(axis=0).mean()),
        }
    return output, coverage


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
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * np.asarray(signal)[active], 0.001, 0.999
    )
    return output, active


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, bank: dict[str, np.ndarray]
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for name, signal in bank.items():
        for route in ROUTES:
            for weight in WEIGHTS:
                candidate, active = _candidate(frame, parent, signal, route, weight)
                result = diagnostics(frame, parent, candidate, active)
                applied_domain = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                output.append(
                    {
                        "signal": name,
                        "domain": route,
                        "weight": float(weight),
                        "gain": float(result["gain"]),
                        "positive_month_fraction": float(result["positive_month_fraction"]),
                        "worst_month_gain": float(result["worst_month_gain"]),
                        "minimum_domain_gain": float(result["minimum_domain_gain"]),
                        "applied_domain_gain": float(applied_domain),
                        "selection_score": float(
                            min(result["gain"], result["worst_month_gain"], applied_domain)
                        ),
                        "mean_abs_shift": float(result["mean_abs_shift"]),
                    }
                )
    return pd.DataFrame(output)


def load_final_parent(
    cache_dir: Path, axis_name: str, frame: pd.DataFrame
) -> np.ndarray:
    """Load and strictly align one exact 1158 parent OOF vector."""

    with np.load(cache_dir / f"{axis_name}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["final_gate_parent"].astype(np.float64)
        month = saved["game_month"].astype(np.int16)
        domain = saved["domain3"].astype(str)
    checks = (
        np.array_equal(target, frame["target"].to_numpy(np.float64)),
        np.array_equal(month, frame["game_month"].to_numpy(np.int16)),
        np.array_equal(domain, frame["domain3"].astype(str).to_numpy()),
    )
    if not all(checks):
        raise ValueError(f"final-parent cache alignment failed: {axis_name} {checks}")
    return parent


def _v50_shift_correlation(
    v50_dir: Path | None, axis_name: str, target: np.ndarray, shift: np.ndarray
) -> float | None:
    if v50_dir is None or not (v50_dir / f"{axis_name}.npz").exists():
        return None
    with np.load(v50_dir / f"{axis_name}.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["target"].astype(np.float64), target):
            raise ValueError(f"v50 target/order mismatch: {axis_name}")
        old_shift = saved["candidate"].astype(np.float64) - saved["v27"].astype(
            np.float64
        )
    if float(np.std(old_shift)) == 0.0 or float(np.std(shift)) == 0.0:
        return 0.0
    return float(np.corrcoef(old_shift, shift)[0, 1])


def run(
    project: Path,
    final_parent_dir: Path,
    output_dir: Path,
    v50_dir: Path | None = None,
) -> dict[str, object]:
    project = project.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir = output_dir.resolve()
    v50_dir = v50_dir.resolve() if v50_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(project)
    models, source_audit = _load_source_models(project, rows)
    banks: dict[int, dict[str, np.ndarray]] = {}
    coverage: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        banks[year], coverage[str(year)] = build_audit_bank(models, rows[year], year)

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
    late23 = rows[2023]["game_month"].ge(8).to_numpy()
    frame23 = axes["selection_late_2023"]
    parent23 = load_final_parent(final_parent_dir, "selection_late_2023", frame23)
    stage2 = _screen(
        frame23,
        parent23,
        {name: value[late23] for name, value in banks[2023].items()},
    )
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    late24 = rows[2024]["game_month"].ge(8).to_numpy()
    audit_signals = {
        "outer_full_2024": banks[2024][recipe["signal"]],
        "replication_late_2024": banks[2024][recipe["signal"]][late24],
    }
    audits: dict[str, dict[str, object]] = {}
    correlations: dict[str, float | None] = {}
    for axis_name, signal in audit_signals.items():
        frame = axes[axis_name]
        parent = load_final_parent(final_parent_dir, axis_name, frame)
        candidate, active = _candidate(
            frame, parent, signal, recipe["domain"], recipe["weight"]
        )
        audits[axis_name] = diagnostics(frame, parent, candidate, active)
        shift = candidate - parent
        correlations[axis_name] = _v50_shift_correlation(
            v50_dir,
            axis_name,
            frame["target"].to_numpy(np.float64),
            shift,
        )
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            final_parent=parent,
            signal=signal,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    outer = audits["outer_full_2024"]
    replication = audits["replication_late_2024"]
    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": float(outer["gain"]) >= 5.0,
        "outer_month_fraction_at_least_075": float(outer["positive_month_fraction"]) >= 0.75,
        "outer_worst_month_above_minus_5": float(outer["worst_month_gain"]) > -5.0,
        "outer_all_domains_nonnegative": all(
            float(value) >= 0.0 for value in outer["domain_gains"].values()
        ),
        "replication_gain_positive": float(replication["gain"]) > 0.0,
        "replication_all_months_positive": float(replication["positive_month_fraction"]) == 1.0,
    }
    summary = {
        "protocol": "V63_TWO_ORIGIN_PITCHER_BALLS_AHEAD_INTERACTION_V1",
        "parent": "exact OOF analogue of standalone Public 1158.0745556751 on late23/2024",
        "selection_2022_parent_note": "historical incumbent used only as the first prefilter origin",
        "candidate_origin": "public pattern independently rebuilt from local train-only OOF residuals",
        "source_model": "pure pitcher x (balls > strikes) contrast with pitcher main effect removed",
        "grid": {
            "smoothing": list(SMOOTHING_GRID),
            "history_modes": list(HISTORY_MODES),
            "routes": list(ROUTES),
            "weights": list(WEIGHTS),
        },
        "source_diagnostics": source_audit,
        "coverage": coverage,
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
        "audits": audits,
        "shift_correlation_with_v50_lowrank": correlations,
        "gates": {name: bool(value) for name, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v50-dir", type=Path)
    args = parser.parse_args()
    run(args.project, args.final_parent_dir, args.output_dir, args.v50_dir)


if __name__ == "__main__":
    main()
