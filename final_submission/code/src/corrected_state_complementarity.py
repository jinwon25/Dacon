"""Check whether the corrected-state residual adds to the prior CB context axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.corrected_state_residual_oof import (
    _candidate_prediction,
    _load_folds,
    _markdown_table,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


YEARS = (2022, 2023, 2024)
BETA_GRID = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
NEW_RECIPE = {
    2022: ("corrected_state", "r_only", 0.25),
    2023: ("corrected_state", "all", 0.50),
    2024: ("corrected_state", "all", 1.00),
}


def _context_path(research_project: Path, year: int) -> Path:
    matches = sorted(
        (research_project / "artifacts" / "top1100" / "r1_gated_context").rglob(
            f"o{year}.npz"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one context artifact for {year}, found {matches}")
    return matches[0]


def _load_context(research_project: Path, fold: pd.DataFrame, year: int) -> np.ndarray:
    with np.load(_context_path(research_project, year), allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        prediction = bundle["prediction"].astype(np.float64)
        target = bundle["target"].astype(np.float64)
        v2 = bundle["v2"].astype(np.float64)
    if not np.array_equal(row_id, fold["row_id"].astype(str).to_numpy()):
        raise ValueError(f"context row order mismatch for {year}")
    if not np.array_equal(target, fold["target"].to_numpy()):
        raise ValueError(f"context target mismatch for {year}")
    if not np.allclose(v2, fold["v2"].to_numpy(), rtol=0, atol=1e-12):
        raise ValueError(f"context V2 mismatch for {year}")
    return prediction


def _new_prediction(correction_dir: Path, fold: pd.DataFrame, year: int) -> np.ndarray:
    recipe, mode, eta = NEW_RECIPE[year]
    path = correction_dir / f"correction_{recipe}_{mode}_o{year}.npz"
    with np.load(path, allow_pickle=True) as bundle:
        correction = bundle["correction"].astype(np.float64)
        row_id = bundle["row_id"].astype(str)
    expected = fold.loc[fold["game_type"].eq("R"), "row_id"].astype(str).to_numpy()
    if not np.array_equal(row_id, expected):
        raise ValueError(f"correction row order mismatch for {year}")
    return _candidate_prediction(fold, correction, eta)


def _choose_beta(metrics: pd.DataFrame, history_years: tuple[int, ...]) -> float:
    history = metrics.loc[metrics["season"].isin(history_years)].copy()
    weights = {year: i + 1 for i, year in enumerate(history_years)}
    history["weight"] = history["season"].map(weights)
    summary = history.groupby("beta").apply(
        lambda group: np.average(group["delta"], weights=group["weight"]),
        include_groups=False,
    )
    best = summary.min()
    return float(min(beta for beta, value in summary.items() if value <= best + 1e-15))


def run(
    data_project: Path,
    research_project: Path,
    correction_dir: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    correction_dir = correction_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)

    new_prediction = {year: _new_prediction(correction_dir, folds[year], year) for year in YEARS}
    context_prediction = {
        year: _load_context(research_project, folds[year], year) for year in YEARS
    }
    rows = []
    for year in YEARS:
        fold = folds[year]
        target = fold["target"].to_numpy()
        v2 = fold["v2"].to_numpy()
        context_effect = context_prediction[year] - v2
        regular = fold["game_type"].eq("R").to_numpy()
        effect_correlation = float(
            np.corrcoef(
                (new_prediction[year] - v2)[regular], context_effect[regular]
            )[0, 1]
        )
        for beta in BETA_GRID:
            prediction = np.clip(new_prediction[year] + beta * context_effect, 1e-6, 1 - 1e-6)
            rows.append(
                {
                    "season": year,
                    "beta": beta,
                    "brier": brier_score(target, prediction),
                    "v2_brier": brier_score(target, v2),
                    "delta": brier_score(target, prediction) - brier_score(target, v2),
                    "new_only_delta": brier_score(target, new_prediction[year])
                    - brier_score(target, v2),
                    "context_only_delta": brier_score(target, context_prediction[year])
                    - brier_score(target, v2),
                    "effect_correlation_r": effect_correlation,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    selections = []
    selected = {}
    for year, history_years in ((2022, ()), (2023, (2022,)), (2024, (2023,))):
        beta = 0.0 if not history_years else _choose_beta(metrics, history_years)
        prediction = np.clip(
            new_prediction[year]
            + beta * (context_prediction[year] - folds[year]["v2"].to_numpy()),
            1e-6,
            1 - 1e-6,
        )
        selected[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], folds[year]["v2"]
        )
        selections.append(
            {
                "season": year,
                "selection_years": ",".join(map(str, history_years)) or "fixed_new_only",
                "context_beta": beta,
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    selection = pd.DataFrame(selections)
    selection.to_csv(out_dir / "latest_origin_selection.csv", index=False)
    target = np.concatenate([folds[year]["target"].to_numpy() for year in YEARS])
    baseline = np.concatenate([folds[year]["v2"].to_numpy() for year in YEARS])
    candidate = np.concatenate([selected[year] for year in YEARS])
    clusters = np.concatenate(
        [
            (str(year) + ":" + folds[year]["pitcher_id"].astype(str)).to_numpy()
            for year in YEARS
        ]
    )
    bootstrap = cluster_bootstrap_delta(
        target,
        candidate,
        baseline,
        clusters,
        n_resamples=n_resamples,
        seed=20260816,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    report = (
        "# Corrected-state and CB-context complementarity\n\n"
        "This is an exploratory combination audit. The CB-context axis was developed "
        "before this cycle and is not a fresh candidate.\n\n"
        + _markdown_table(selection)
        + f"\n\nRecency-weighted delta: `{weighted_delta:.12f}`.\n\n"
        + "```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n"
    )
    (out_dir / "decision.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(selection.to_string(index=False))
    print(f"recency_weighted_delta={weighted_delta:.12f}")
    print(json.dumps(bootstrap, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.data_project,
        args.research_project,
        args.correction_dir,
        args.out_dir,
        args.n_resamples,
    )


if __name__ == "__main__":
    main()
