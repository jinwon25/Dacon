"""Forward residual OOF using corrected state plus entity interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.corrected_state_residual_oof import (
    _candidate_prediction,
    _choose_recipe,
    _fit_correction,
    _load_folds,
    _markdown_table,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta
from src.top1100_features import build_features


YEARS = (2022, 2023, 2024)
ETAS = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 1.00, 1.25)


def _string(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("__MISSING__")


def _features(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    features = build_features(train, train, include_ids=True)
    count = _string(train, "balls_before") + "-" + _string(train, "strikes_before")
    features["pitcher_count"] = (_string(train, "pitcher_id") + "|" + count).astype(
        "category"
    )
    features["batter_count"] = (_string(train, "batter_id") + "|" + count).astype(
        "category"
    )
    features["pitcher_batter_hand"] = (
        _string(train, "pitcher_id") + "|" + _string(train, "batter_hand")
    ).astype("category")
    features["pitcher_team_count"] = (
        _string(train, "pitcher_team_id") + "|" + count
    ).astype("category")
    features["team_matchup"] = (
        _string(train, "pitcher_team_id") + "|" + _string(train, "batter_team_id")
    ).astype("category")
    interaction_columns = [
        "pitcher_count",
        "batter_count",
        "pitcher_batter_hand",
        "pitcher_team_count",
        "team_matchup",
    ]
    return features, {
        "state_ids": [column for column in features.columns if column not in interaction_columns],
        "state_id_interactions": list(features.columns),
    }


def run(
    data_project: Path,
    research_project: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)
    features, recipes = _features(train)

    rows: list[dict[str, object]] = []
    corrections: dict[tuple[int, str], np.ndarray] = {}
    for year in YEARS:
        history = pd.concat(
            [folds[past] for past in (2021, 2022, 2023) if past < year],
            ignore_index=True,
        )
        audit = folds[year]
        regular = audit["game_type"].eq("R").to_numpy()
        fit_index = history["train_index"].to_numpy(np.int64)
        audit_index = audit.loc[regular, "train_index"].to_numpy(np.int64)
        fit_residual = history["target"].to_numpy() - history["v2"].to_numpy()
        for recipe, columns in recipes.items():
            correction = _fit_correction(
                features.iloc[fit_index][columns].reset_index(drop=True),
                fit_residual,
                features.iloc[audit_index][columns].reset_index(drop=True),
            )
            corrections[(year, recipe)] = correction
            np.savez_compressed(
                out_dir / f"correction_{recipe}_o{year}.npz",
                row_id=audit.loc[regular, "row_id"].astype(str).to_numpy(),
                correction=correction,
            )
            baseline = brier_score(audit["target"], audit["v2"])
            for eta in ETAS:
                prediction = _candidate_prediction(audit, correction, eta)
                rows.append(
                    {
                        "season": year,
                        "feature_recipe": recipe,
                        "fit_mode": "ALL",
                        "eta": eta,
                        "delta": brier_score(audit["target"], prediction) - baseline,
                        "brier": brier_score(audit["target"], prediction),
                        "v2_brier": baseline,
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    fixed_o22 = pd.Series(
        {
            "feature_recipe": "state_ids",
            "fit_mode": "ALL",
            "eta": 0.25,
            "selection_delta": np.nan,
        }
    )
    selections = []
    predictions = {}
    for year, history_years in ((2022, ()), (2023, (2022,)), (2024, (2023,))):
        choice = fixed_o22 if not history_years else _choose_recipe(metrics, history_years)
        recipe = str(choice["feature_recipe"])
        eta = float(choice["eta"])
        prediction = _candidate_prediction(
            fold=folds[year], correction=corrections[(year, recipe)], eta=eta
        )
        predictions[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], folds[year]["v2"]
        )
        selections.append(
            {
                "season": year,
                "selection_years": ",".join(map(str, history_years)) or "preregistered",
                "feature_recipe": recipe,
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
        seed=20260817,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (out_dir / "decision.md").write_text(
        "# Entity-state residual OOF\n\n"
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
                "eta_grid": ETAS,
                "feature_counts": {name: len(value) for name, value in recipes.items()},
                "outer_target_used_for_fit": False,
                "outer_target_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selection.to_string(index=False), flush=True)
    print(f"recency_weighted_delta={weighted_delta:.12f}", flush=True)
    print(json.dumps(bootstrap, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(args.data_project, args.research_project, args.out_dir, args.n_resamples)


if __name__ == "__main__":
    main()
