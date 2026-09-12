"""Compare all-history and recent-history corrected-state residual learners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.corrected_state_residual_oof import (
    _candidate_prediction,
    _fit_correction,
    _load_folds,
    _make_features,
    _markdown_table,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


YEARS = (2022, 2023, 2024)
ETAS = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 1.00, 1.25)
WINDOWS = ("ALL", "LAST2", "LAST1")


def _source_correction(source_dir: Path, year: int) -> np.ndarray:
    path = source_dir / f"correction_corrected_state_all_o{year}.npz"
    with np.load(path, allow_pickle=True) as bundle:
        return bundle["correction"].astype(np.float64)


def _history_years(year: int, window: str) -> tuple[int, ...]:
    available = tuple(past for past in (2021, 2022, 2023) if past < year)
    if window == "LAST1":
        return available[-1:]
    if window == "LAST2":
        return available[-2:]
    return available


def _choose(metrics: pd.DataFrame, history_years: tuple[int, ...]) -> pd.Series:
    frame = metrics.loc[metrics["season"].isin(history_years)].copy()
    frame["weight"] = frame["season"].map(
        {year: i + 1 for i, year in enumerate(history_years)}
    )
    summary = (
        frame.groupby(["window", "eta"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "selection_delta": np.average(
                        group["delta"], weights=group["weight"]
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    summary["window_order"] = summary["window"].map({"ALL": 0, "LAST2": 1, "LAST1": 2})
    return summary.sort_values(
        ["selection_delta", "eta", "window_order"], kind="mergesort"
    ).iloc[0]


def run(
    data_project: Path,
    research_project: Path,
    source_dir: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)
    features = _make_features(train)

    corrections: dict[tuple[int, str], np.ndarray] = {}
    for year in YEARS:
        corrections[(year, "ALL")] = _source_correction(source_dir, year)
        for window in ("LAST2", "LAST1"):
            history_years = _history_years(year, window)
            all_years = _history_years(year, "ALL")
            if history_years == all_years:
                corrections[(year, window)] = corrections[(year, "ALL")]
                continue
            history = pd.concat([folds[past] for past in history_years], ignore_index=True)
            audit = folds[year]
            regular = audit["game_type"].eq("R").to_numpy()
            correction = _fit_correction(
                features.iloc[history["train_index"].to_numpy()].reset_index(drop=True),
                history["target"].to_numpy() - history["v2"].to_numpy(),
                features.iloc[
                    audit.loc[regular, "train_index"].to_numpy()
                ].reset_index(drop=True),
            )
            corrections[(year, window)] = correction
            np.savez_compressed(
                out_dir / f"correction_{window.lower()}_o{year}.npz",
                row_id=audit.loc[regular, "row_id"].astype(str).to_numpy(),
                correction=correction,
                history_years=np.asarray(history_years),
            )

    rows = []
    for year in YEARS:
        fold = folds[year]
        baseline = brier_score(fold["target"], fold["v2"])
        for window in WINDOWS:
            for eta in ETAS:
                prediction = _candidate_prediction(fold, corrections[(year, window)], eta)
                rows.append(
                    {
                        "season": year,
                        "window": window,
                        "eta": eta,
                        "delta": brier_score(fold["target"], prediction) - baseline,
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    fixed_o22 = pd.Series(
        {"window": "ALL", "eta": 0.25, "selection_delta": np.nan}
    )
    selections = []
    predictions = {}
    for year, selection_years in ((2022, ()), (2023, (2022,)), (2024, (2023,))):
        choice = fixed_o22 if not selection_years else _choose(metrics, selection_years)
        window = str(choice["window"])
        eta = float(choice["eta"])
        prediction = _candidate_prediction(folds[year], corrections[(year, window)], eta)
        predictions[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], folds[year]["v2"]
        )
        selections.append(
            {
                "season": year,
                "selection_years": ",".join(map(str, selection_years)) or "preregistered",
                "window": window,
                "eta": eta,
                "selection_delta": float(choice["selection_delta"]),
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
        seed=20260818,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (out_dir / "decision.md").write_text(
        "# Recent-window corrected-state residual OOF\n\n"
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
                "windows": WINDOWS,
                "eta_grid": ETAS,
                "source_all_history_corrections": str(source_dir),
                "outer_target_used_for_fit": False,
                "outer_target_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)
    print(selection.to_string(index=False), flush=True)
    print(f"recency_weighted_delta={weighted_delta:.12f}", flush=True)
    print(json.dumps(bootstrap, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.data_project,
        args.research_project,
        args.source_dir,
        args.out_dir,
        args.n_resamples,
    )


if __name__ == "__main__":
    main()
