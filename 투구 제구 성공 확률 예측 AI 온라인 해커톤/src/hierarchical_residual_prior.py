"""Forward empirical-Bayes residual priors on top of corrected-state predictions."""

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
BASE_ETA = {2022: 0.25, 2023: 0.50, 2024: 1.00}
K_VALUES = (100.0, 300.0, 1000.0, 3000.0)
ETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
KEYS = {
    "pitcher": ("pitcher_id",),
    "pitcher_count": ("pitcher_id", "count_state"),
    "pitcher_platoon": ("pitcher_id", "batter_hand"),
    "batter": ("batter_id",),
    "pitcher_team_count": ("pitcher_team_id", "count_state"),
    "count_platoon": ("count_state", "pitcher_hand", "batter_hand"),
}


def _load_base(correction_dir: Path, fold: pd.DataFrame, year: int) -> np.ndarray:
    mode = "r_only" if year == 2022 else "all"
    path = correction_dir / f"correction_corrected_state_{mode}_o{year}.npz"
    with np.load(path, allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        correction = bundle["correction"].astype(np.float64)
    expected = fold.loc[fold["game_type"].eq("R"), "row_id"].astype(str).to_numpy()
    if not np.array_equal(row_id, expected):
        raise ValueError(f"base correction row mismatch for {year}")
    return _candidate_prediction(fold, correction, BASE_ETA[year])


def _metadata(train: pd.DataFrame, fold: pd.DataFrame) -> pd.DataFrame:
    rows = train.iloc[fold["train_index"].to_numpy()].reset_index(drop=True)
    return pd.DataFrame(
        {
            "pitcher_id": rows["pitcher_id"].astype(str),
            "batter_id": rows["batter_id"].astype(str),
            "pitcher_team_id": rows["pitcher_team_id"].astype(str),
            "pitcher_hand": rows["pitcher_hand"].astype(str),
            "batter_hand": rows["batter_hand"].astype(str),
            "count_state": rows["balls_before"].astype(str)
            + "-"
            + rows["strikes_before"].astype(str),
            "game_type": rows["game_type"].astype(str),
        }
    )


def _prior(
    history_meta: pd.DataFrame,
    history_residual: np.ndarray,
    audit_meta: pd.DataFrame,
    columns: tuple[str, ...],
    k: float,
) -> np.ndarray:
    history = history_meta.loc[:, list(columns)].copy()
    history["residual"] = history_residual
    grouped = history.groupby(list(columns), observed=True)["residual"].agg(["sum", "size"])
    query = audit_meta.loc[:, list(columns)].copy()
    if len(columns) == 1:
        key = query[columns[0]]
        sums = key.map(grouped["sum"]).fillna(0.0).to_numpy(np.float64)
        counts = key.map(grouped["size"]).fillna(0.0).to_numpy(np.float64)
    else:
        index = pd.MultiIndex.from_frame(query)
        sums = grouped["sum"].reindex(index).fillna(0.0).to_numpy(np.float64)
        counts = grouped["size"].reindex(index).fillna(0.0).to_numpy(np.float64)
    return sums / (counts + k)


def _choose(metrics: pd.DataFrame, year: int) -> pd.Series:
    frame = metrics.loc[metrics["season"].eq(year)].copy()
    best = frame["delta"].min()
    frame = frame.loc[frame["delta"].le(best + 1e-15)]
    return frame.sort_values(["delta", "eta", "k", "key"], kind="mergesort").iloc[0]


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
    meta = {year: _metadata(train, folds[year]) for year in (2021, 2022, 2023, 2024)}
    base = {2021: folds[2021]["v2"].to_numpy()}
    base.update({year: _load_base(correction_dir, folds[year], year) for year in YEARS})

    priors: dict[tuple[int, str, float], np.ndarray] = {}
    rows = []
    for year in (2023, 2024):
        history_years = tuple(past for past in YEARS if past < year)
        history_meta = pd.concat([meta[past] for past in history_years], ignore_index=True)
        history_residual = np.concatenate(
            [folds[past]["target"].to_numpy() - base[past] for past in history_years]
        )
        history_regular = history_meta["game_type"].eq("R").to_numpy()
        history_meta = history_meta.loc[history_regular].reset_index(drop=True)
        history_residual = history_residual[history_regular]
        audit_regular = meta[year]["game_type"].eq("R").to_numpy()
        baseline_brier = brier_score(folds[year]["target"], folds[year]["v2"])
        for key, columns in KEYS.items():
            for k in K_VALUES:
                correction = _prior(
                    history_meta,
                    history_residual,
                    meta[year].loc[audit_regular].reset_index(drop=True),
                    columns,
                    k,
                )
                priors[(year, key, k)] = correction
                for eta in ETAS:
                    prediction = base[year].copy()
                    prediction[audit_regular] = np.clip(
                        prediction[audit_regular] + eta * correction, 1e-6, 1 - 1e-6
                    )
                    rows.append(
                        {
                            "season": year,
                            "key": key,
                            "k": k,
                            "eta": eta,
                            "delta": brier_score(folds[year]["target"], prediction)
                            - baseline_brier,
                            "base_delta": brier_score(folds[year]["target"], base[year])
                            - baseline_brier,
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    selections = [
        {
            "season": 2022,
            "selection_year": "fixed_base_only",
            "key": "none",
            "k": np.nan,
            "eta": 0.0,
            "audit_delta": brier_score(folds[2022]["target"], base[2022])
            - brier_score(folds[2022]["target"], folds[2022]["v2"]),
            "outer_target_used_for_selection": False,
        }
    ]
    predictions = {2022: base[2022]}
    # O23 has no earlier comparable prior audit, so it remains base-only.
    selections.append(
        {
            "season": 2023,
            "selection_year": "fixed_base_only",
            "key": "none",
            "k": np.nan,
            "eta": 0.0,
            "audit_delta": brier_score(folds[2023]["target"], base[2023])
            - brier_score(folds[2023]["target"], folds[2023]["v2"]),
            "outer_target_used_for_selection": False,
        }
    )
    predictions[2023] = base[2023]
    choice = _choose(metrics, 2023)
    key = str(choice["key"])
    k = float(choice["k"])
    eta = float(choice["eta"])
    regular = meta[2024]["game_type"].eq("R").to_numpy()
    prediction = base[2024].copy()
    prediction[regular] = np.clip(
        prediction[regular] + eta * priors[(2024, key, k)], 1e-6, 1 - 1e-6
    )
    predictions[2024] = prediction
    selections.append(
        {
            "season": 2024,
            "selection_year": 2023,
            "key": key,
            "k": k,
            "eta": eta,
            "audit_delta": brier_score(folds[2024]["target"], prediction)
            - brier_score(folds[2024]["target"], folds[2024]["v2"]),
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
        seed=20260821,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (out_dir / "decision.md").write_text(
        "# Hierarchical residual prior\n\n"
        + _markdown_table(selection)
        + f"\n\nRecency-weighted delta: `{weighted_delta:.12f}`.\n\n"
        + "```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
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
