"""Forward ensemble audit for the three surviving corrected-state axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.corrected_state_complementarity import _load_context
from src.corrected_state_residual_oof import (
    _candidate_prediction,
    _load_folds,
    _markdown_table,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


YEARS = (2022, 2023, 2024)
AXIS_WEIGHTS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75)
RESIDUAL_ETA = {2022: 0.25, 2023: 0.50, 2024: 1.00}
CB_WEIGHT = {2022: 0.50, 2023: 0.75, 2024: 0.75}


def _load_residual(correction_dir: Path, fold: pd.DataFrame, year: int) -> np.ndarray:
    mode = "r_only" if year == 2022 else "all"
    path = correction_dir / f"correction_corrected_state_{mode}_o{year}.npz"
    with np.load(path, allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        correction = bundle["correction"].astype(np.float64)
    expected = fold.loc[fold["game_type"].eq("R"), "row_id"].astype(str).to_numpy()
    if not np.array_equal(row_id, expected):
        raise ValueError(f"residual row mismatch for {year}")
    return _candidate_prediction(fold, correction, RESIDUAL_ETA[year])


def _load_corrected_cb(cb_dir: Path, fold: pd.DataFrame, year: int) -> np.ndarray:
    with np.load(cb_dir / f"corrected_cb_o{year}.npz", allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        cb = bundle["prediction"].astype(np.float64)
    if not np.array_equal(row_id, fold["row_id"].astype(str).to_numpy()):
        raise ValueError(f"corrected CB row mismatch for {year}")
    prediction = fold["v2"].to_numpy(copy=True)
    regular = fold["game_type"].eq("R").to_numpy()
    weight = CB_WEIGHT[year]
    prediction[regular] = (1 - weight) * prediction[regular] + weight * cb[regular]
    return prediction


def _choose(metrics: pd.DataFrame, year: int) -> pd.Series:
    candidates = metrics.loc[metrics["season"].eq(year)].copy()
    best = candidates["delta"].min()
    candidates = candidates.loc[candidates["delta"].le(best + 1e-15)]
    candidates["complexity"] = candidates["cb_axis_weight"] + candidates["context_axis_weight"]
    return candidates.sort_values(
        ["delta", "complexity", "cb_axis_weight", "context_axis_weight"],
        kind="mergesort",
    ).iloc[0]


def run(
    data_project: Path,
    research_project: Path,
    correction_dir: Path,
    corrected_cb_dir: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    correction_dir = correction_dir.resolve()
    corrected_cb_dir = corrected_cb_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)
    residual = {year: _load_residual(correction_dir, folds[year], year) for year in YEARS}
    corrected_cb = {
        year: _load_corrected_cb(corrected_cb_dir, folds[year], year) for year in YEARS
    }
    context = {year: _load_context(research_project, folds[year], year) for year in YEARS}

    rows = []
    correlations = []
    for year in YEARS:
        fold = folds[year]
        target = fold["target"].to_numpy()
        v2 = fold["v2"].to_numpy()
        regular = fold["game_type"].eq("R").to_numpy()
        effects = np.vstack(
            [
                (residual[year] - v2)[regular],
                (corrected_cb[year] - v2)[regular],
                (context[year] - v2)[regular],
            ]
        )
        corr = np.corrcoef(effects)
        correlations.append(
            {
                "season": year,
                "residual_cb": corr[0, 1],
                "residual_context": corr[0, 2],
                "cb_context": corr[1, 2],
            }
        )
        baseline = brier_score(target, v2)
        for cb_weight in AXIS_WEIGHTS:
            for context_weight in AXIS_WEIGHTS:
                prediction = np.clip(
                    residual[year]
                    + cb_weight * (corrected_cb[year] - v2)
                    + context_weight * (context[year] - v2),
                    1e-6,
                    1 - 1e-6,
                )
                rows.append(
                    {
                        "season": year,
                        "cb_axis_weight": cb_weight,
                        "context_axis_weight": context_weight,
                        "delta": brier_score(target, prediction) - baseline,
                        "residual_only_delta": brier_score(target, residual[year]) - baseline,
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)
    pd.DataFrame(correlations).to_csv(out_dir / "effect_correlations.csv", index=False)

    selections = []
    predictions = {}
    for year, selection_year in ((2022, None), (2023, 2022), (2024, 2023)):
        if selection_year is None:
            choice = pd.Series(
                {"cb_axis_weight": 0.0, "context_axis_weight": 0.0, "delta": np.nan}
            )
        else:
            choice = _choose(metrics, selection_year)
        cb_weight = float(choice["cb_axis_weight"])
        context_weight = float(choice["context_axis_weight"])
        v2 = folds[year]["v2"].to_numpy()
        prediction = np.clip(
            residual[year]
            + cb_weight * (corrected_cb[year] - v2)
            + context_weight * (context[year] - v2),
            1e-6,
            1 - 1e-6,
        )
        predictions[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], v2
        )
        selections.append(
            {
                "season": year,
                "selection_year": selection_year or "fixed_residual_only",
                "cb_axis_weight": cb_weight,
                "context_axis_weight": context_weight,
                "selection_delta": float(choice["delta"]),
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    selection = pd.DataFrame(selections)
    selection.to_csv(out_dir / "latest_origin_selection.csv", index=False)
    target = np.concatenate([folds[year]["target"].to_numpy() for year in YEARS])
    baseline = np.concatenate([folds[year]["v2"].to_numpy() for year in YEARS])
    candidate = np.concatenate([predictions[year] for year in YEARS])
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
        seed=20260820,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (out_dir / "decision.md").write_text(
        "# Corrected-family forward ensemble\n\n"
        + _markdown_table(selection)
        + f"\n\nRecency-weighted delta: `{weighted_delta:.12f}`.\n\n"
        + "```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "axis_grid": AXIS_WEIGHTS,
                "residual_eta": RESIDUAL_ETA,
                "corrected_cb_weight": CB_WEIGHT,
                "outer_target_used_for_selection": False,
                "note": "old context axis is exploratory and was developed before this cycle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(correlations).to_string(index=False))
    print(selection.to_string(index=False))
    print(f"recency_weighted_delta={weighted_delta:.12f}")
    print(json.dumps(bootstrap, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, required=True)
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.data_project,
        args.research_project,
        args.correction_dir,
        args.corrected_cb_dir,
        args.out_dir,
        args.n_resamples,
    )


if __name__ == "__main__":
    main()
