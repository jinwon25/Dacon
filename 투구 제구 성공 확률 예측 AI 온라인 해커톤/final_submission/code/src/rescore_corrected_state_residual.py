"""Rescore persisted corrected-state residuals on an expanded eta grid."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.corrected_state_residual_oof import (
    _candidate_prediction,
    _choose_recipe,
    _load_folds,
    _markdown_table,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


YEARS = (2022, 2023, 2024)
ETAS = (0.0, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 1.00, 1.25)
NAME = re.compile(r"correction_(context|corrected_state)_(all|r_only)_o(\d{4})\.npz")


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
    corrections: dict[tuple[int, str, str], np.ndarray] = {}
    for path in correction_dir.glob("correction_*.npz"):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        recipe, mode_token, year_token = match.groups()
        with np.load(path, allow_pickle=True) as bundle:
            correction = bundle["correction"].astype(np.float64)
            row_id = bundle["row_id"].astype(str)
        year = int(year_token)
        expected = folds[year].loc[folds[year]["game_type"].eq("R"), "row_id"].astype(str)
        if not np.array_equal(row_id, expected.to_numpy()):
            raise ValueError(f"row order mismatch in {path}")
        corrections[(year, recipe, mode_token.upper())] = correction

    rows: list[dict[str, object]] = []
    for (year, recipe, mode), correction in corrections.items():
        fold = folds[year]
        baseline = brier_score(fold["target"], fold["v2"])
        regular = fold["game_type"].eq("R").to_numpy()
        r_baseline = brier_score(
            fold.loc[regular, "target"], fold.loc[regular, "v2"]
        )
        for eta in ETAS:
            prediction = _candidate_prediction(fold, correction, eta)
            rows.append(
                {
                    "season": year,
                    "feature_recipe": recipe,
                    "fit_mode": mode,
                    "eta": eta,
                    "brier": brier_score(fold["target"], prediction),
                    "v2_brier": baseline,
                    "delta": brier_score(fold["target"], prediction) - baseline,
                    "r_delta": brier_score(fold.loc[regular, "target"], prediction[regular])
                    - r_baseline,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "expanded_grid_metrics.csv", index=False)

    fixed_o22 = pd.Series(
        {
            "feature_recipe": "corrected_state",
            "fit_mode": "R_ONLY",
            "eta": 0.25,
            "selection_delta": np.nan,
        }
    )
    selections: list[dict[str, object]] = []
    predictions: dict[int, np.ndarray] = {}
    for year, history_years in ((2022, ()), (2023, (2022,)), (2024, (2022, 2023))):
        choice = fixed_o22 if not history_years else _choose_recipe(metrics, history_years)
        recipe = str(choice["feature_recipe"])
        mode = str(choice["fit_mode"])
        eta = float(choice["eta"])
        prediction = _candidate_prediction(
            folds[year], corrections[(year, recipe, mode)], eta
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
                "fit_mode": mode,
                "eta": eta,
                "selection_delta": float(choice["selection_delta"]),
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    selection = pd.DataFrame(selections)
    selection.to_csv(out_dir / "nested_selection.csv", index=False)

    latest_selections: list[dict[str, object]] = []
    latest_predictions: dict[int, np.ndarray] = {}
    for year, history_years in ((2022, ()), (2023, (2022,)), (2024, (2023,))):
        choice = fixed_o22 if not history_years else _choose_recipe(metrics, history_years)
        recipe = str(choice["feature_recipe"])
        mode = str(choice["fit_mode"])
        eta = float(choice["eta"])
        prediction = _candidate_prediction(
            folds[year], corrections[(year, recipe, mode)], eta
        )
        latest_predictions[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], folds[year]["v2"]
        )
        latest_selections.append(
            {
                "season": year,
                "selection_years": ",".join(map(str, history_years)) or "preregistered",
                "feature_recipe": recipe,
                "fit_mode": mode,
                "eta": eta,
                "selection_delta": float(choice["selection_delta"]),
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    latest_selection = pd.DataFrame(latest_selections)
    latest_selection.to_csv(out_dir / "latest_origin_selection.csv", index=False)

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
        seed=20260814,
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    latest_candidate = np.concatenate([latest_predictions[year] for year in YEARS])
    latest_bootstrap = cluster_bootstrap_delta(
        target,
        latest_candidate,
        baseline,
        clusters,
        n_resamples=n_resamples,
        seed=20260814,
    )
    (out_dir / "latest_origin_bootstrap.json").write_text(
        json.dumps(latest_bootstrap, indent=2), encoding="utf-8"
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=selection["season"] - 2020)
    )
    latest_weighted_delta = float(
        np.average(
            latest_selection["audit_delta"],
            weights=latest_selection["season"] - 2020,
        )
    )

    subgroup_frames = []
    for year in YEARS:
        fold = folds[year]
        source = train.iloc[fold["train_index"].to_numpy()].reset_index(drop=True)
        row_delta = (
            np.square(latest_predictions[year] - fold["target"].to_numpy())
            - np.square(fold["v2"].to_numpy() - fold["target"].to_numpy())
        )
        audit = pd.DataFrame(
            {
                "season": year,
                "game_month": source["game_month"].to_numpy(),
                "count_state": source["balls_before"].astype(str)
                + "-"
                + source["strikes_before"].astype(str),
                "history_bucket": pd.cut(
                    pd.to_numeric(source["asof_pitcher_n"], errors="coerce"),
                    bins=[-np.inf, 30, 100, 300, 1000, np.inf],
                    labels=["0-30", "31-100", "101-300", "301-1000", "1001+"],
                ).astype(str),
                "row_delta": row_delta,
            }
        )
        for dimension in ("season", "game_month", "count_state", "history_bucket"):
            grouped = audit.groupby(dimension, observed=True)["row_delta"].agg(
                ["size", "mean"]
            )
            grouped = grouped.reset_index().rename(
                columns={dimension: "value", "size": "n", "mean": "delta"}
            )
            grouped.insert(0, "dimension", dimension)
            grouped.insert(0, "audit_season", year)
            subgroup_frames.append(grouped)
    subgroups = pd.concat(subgroup_frames, ignore_index=True)
    subgroups.to_csv(out_dir / "latest_origin_subgroups.csv", index=False)

    fold_2024 = folds[2024]
    o24_bootstrap = cluster_bootstrap_delta(
        fold_2024["target"],
        latest_predictions[2024],
        fold_2024["v2"],
        fold_2024["pitcher_id"],
        n_resamples=n_resamples,
        seed=20260815,
    )
    (out_dir / "latest_origin_o24_bootstrap.json").write_text(
        json.dumps(o24_bootstrap, indent=2), encoding="utf-8"
    )
    report = (
        "# Expanded eta rescore\n\n"
        "No model was refit. Corrections are loaded from the source run and only "
        "their scalar strength is rescored. O23/O24 choices use prior outer folds.\n\n"
        + _markdown_table(selection)
        + f"\n\nRecency-weighted selected delta: `{weighted_delta:.12f}`.\n\n"
        + "## Latest-origin policy\n\n"
        + _markdown_table(latest_selection)
        + f"\n\nRecency-weighted selected delta: `{latest_weighted_delta:.12f}`.\n\n"
        + "```json\n"
        + json.dumps(latest_bootstrap, indent=2)
        + "\n```\n"
    )
    (out_dir / "decision.md").write_text(report, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_correction_dir": str(correction_dir),
                "eta_grid": ETAS,
                "models_refit": False,
                "outer_target_used_for_selection": False,
                "latest_origin_weighted_delta": latest_weighted_delta,
                "latest_origin_bootstrap": latest_bootstrap,
                "latest_origin_o24_bootstrap": o24_bootstrap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selection.to_string(index=False))
    print(f"recency_weighted_delta={weighted_delta:.12f}")
    print(latest_selection.to_string(index=False))
    print(f"latest_origin_weighted_delta={latest_weighted_delta:.12f}")
    print(json.dumps(latest_bootstrap, indent=2))
    print(json.dumps({"o24": o24_bootstrap}, indent=2))


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
