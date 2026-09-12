"""Focused paired-bootstrap audit for the retained damped-drift signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset, forecast_base_rate
from src.archive.data import TARGET_COL, read_main
from src.archive.followup import _load_all_caches
from src.metrics import cluster_bootstrap_delta, probability_logit


def _damped_prediction(
    train: pd.DataFrame,
    train_idx: np.ndarray,
    cache: dict[str, np.ndarray],
    year: int,
) -> np.ndarray:
    rates = (
        train.iloc[train_idx]
        .groupby("season", observed=True)[TARGET_COL]
        .mean()
        .sort_index()
    )
    if len(rates) < 3:
        return cache["blend_raw"].astype(np.float64)
    forecast = forecast_base_rate(rates, year, "damped_3_0.8")
    latest = float(rates.iloc[-1])
    offset = float(
        probability_logit(np.array([forecast]))[0]
        - probability_logit(np.array([latest]))[0]
    )
    return apply_logit_offset(cache["blend_raw"], offset)


def run(project_dir: Path, n_resamples: int = 10_000) -> pd.DataFrame:
    config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, config, train)
    years = [int(value) for value in config["outer_validation_seasons"]]
    predictions: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for year in years:
        fold = folds[year]
        cache = caches[year]
        prediction = _damped_prediction(train, fold.train_idx, cache, year)
        predictions[year] = prediction
        valid = train.iloc[fold.valid_idx]
        cluster_specs = {
            "pitcher_season": (
                valid["pitcher_id"].astype("string")
                + "-"
                + valid["season"].astype("string")
            ),
            "pitcher": valid["pitcher_id"].astype("string"),
        }
        for cluster_type, clusters in cluster_specs.items():
            result = cluster_bootstrap_delta(
                cache["target"],
                prediction,
                cache["incumbent"],
                clusters,
                n_resamples=n_resamples,
                seed=int(config["seed"]) + 1_000 + year,
            )
            rows.append(
                {
                    "scope": str(year),
                    "candidate": "damped_3_0.8_fallback_raw",
                    "reference": "incumbent",
                    "cluster_type": cluster_type,
                    **result,
                }
            )

    target = np.concatenate([caches[year]["target"] for year in years])
    candidate = np.concatenate([predictions[year] for year in years])
    incumbent = np.concatenate([caches[year]["incumbent"] for year in years])
    valid_all = pd.concat(
        [train.iloc[folds[year].valid_idx] for year in years], ignore_index=True
    )
    combined_clusters = {
        "pitcher_season": (
            valid_all["pitcher_id"].astype("string")
            + "-"
            + valid_all["season"].astype("string")
        ),
        "pitcher": valid_all["pitcher_id"].astype("string"),
    }
    for cluster_type, clusters in combined_clusters.items():
        result = cluster_bootstrap_delta(
            target,
            candidate,
            incumbent,
            clusters,
            n_resamples=n_resamples,
            seed=int(config["seed"]) + 9_000,
        )
        rows.append(
            {
                "scope": "2021-2024",
                "candidate": "damped_3_0.8_fallback_raw",
                "reference": "incumbent",
                "cluster_type": cluster_type,
                **result,
            }
        )

    result_frame = pd.DataFrame(rows)
    output = project_dir / "research" / "reports" / "damped_bootstrap_audit.csv"
    result_frame.to_csv(output, index=False)
    return result_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    result = run(args.project_dir.resolve(), args.resamples)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
