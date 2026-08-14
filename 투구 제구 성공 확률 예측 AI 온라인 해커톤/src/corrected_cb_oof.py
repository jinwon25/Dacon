"""Rebuild the frozen CB-R1 family after correcting season-state baselines."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.corrected_state_residual_oof import _load_folds, _markdown_table
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta
from src.top1100_features import build_features


YEARS = (2021, 2022, 2023, 2024)
WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.75, 1.0)


def _clean_categories(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    categorical = list(
        frame.select_dtypes(include=["category", "object", "string"]).columns
    )
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame, categorical


def _fit_predict(
    fit: pd.DataFrame,
    target: np.ndarray,
    audit: pd.DataFrame,
) -> tuple[CatBoostClassifier, np.ndarray]:
    fit, categorical = _clean_categories(fit)
    audit, _ = _clean_categories(audit)
    model = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=30,
        random_strength=0.5,
        random_seed=42,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        one_hot_max_size=32,
    )
    model.fit(fit, target, cat_features=categorical)
    prediction = np.asarray(model.predict_proba(audit)[:, 1], dtype=np.float64)
    return model, prediction


def _gated(fold: pd.DataFrame, cb: np.ndarray, weight: float) -> np.ndarray:
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    regular = fold["game_type"].eq("R").to_numpy()
    prediction[regular] = (
        (1.0 - weight) * prediction[regular] + weight * cb[regular]
    )
    return np.clip(prediction, 1e-6, 1 - 1e-6)


def _choose_weight(metrics: pd.DataFrame, year: int) -> float:
    history = metrics.loc[metrics["season"].eq(year)]
    best = history["delta"].min()
    return float(history.loc[history["delta"].le(best + 1e-15), "weight"].min())


def run(
    data_project: Path,
    research_project: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    started = time.perf_counter()
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(research_project, train)
    features = build_features(train, train, include_ids=True)

    predictions: dict[int, np.ndarray] = {}
    fit_rows: dict[int, int] = {}
    fold_seconds: dict[int, float] = {}
    for year in YEARS:
        fold_start = time.perf_counter()
        fit_index = np.flatnonzero(train["season"].to_numpy() < year)
        if len(fit_index) > 1_200_000:
            rng = np.random.default_rng(42)
            fit_index = np.sort(rng.choice(fit_index, size=1_200_000, replace=False))
        audit_index = folds[year]["train_index"].to_numpy(np.int64)
        model, prediction = _fit_predict(
            features.iloc[fit_index].reset_index(drop=True),
            train.iloc[fit_index]["control_success"].to_numpy(np.int8),
            features.iloc[audit_index].reset_index(drop=True),
        )
        predictions[year] = prediction
        fit_rows[year] = len(fit_index)
        fold_seconds[year] = time.perf_counter() - fold_start
        np.savez_compressed(
            out_dir / f"corrected_cb_o{year}.npz",
            row_id=folds[year]["row_id"].astype(str).to_numpy(),
            target=folds[year]["target"].to_numpy(),
            v2=folds[year]["v2"].to_numpy(),
            prediction=prediction,
            game_type=folds[year]["game_type"].astype(str).to_numpy(),
        )
        model.save_model(str(out_dir / f"corrected_cb_o{year}.cbm"))
        del model
        gc.collect()
        print(
            f"year={year} fit_rows={len(fit_index)} seconds={fold_seconds[year]:.1f}",
            flush=True,
        )

    rows = []
    for year in YEARS:
        fold = folds[year]
        target = fold["target"].to_numpy()
        baseline = brier_score(target, fold["v2"])
        regular = fold["game_type"].eq("R").to_numpy()
        raw_r_delta = brier_score(target[regular], predictions[year][regular]) - brier_score(
            target[regular], fold.loc[regular, "v2"]
        )
        raw_f_delta = brier_score(target[~regular], predictions[year][~regular]) - brier_score(
            target[~regular], fold.loc[~regular, "v2"]
        )
        for weight in WEIGHTS:
            prediction = _gated(fold, predictions[year], weight)
            rows.append(
                {
                    "season": year,
                    "weight": weight,
                    "delta": brier_score(target, prediction) - baseline,
                    "raw_r_delta": raw_r_delta,
                    "raw_f_delta": raw_f_delta,
                    "n_fit": fit_rows[year],
                    "seconds": fold_seconds[year],
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "gated_grid_metrics.csv", index=False)

    selection_rows = []
    selected = {}
    for year, selection_year in ((2021, None), (2022, 2021), (2023, 2022), (2024, 2023)):
        weight = 0.0 if selection_year is None else _choose_weight(metrics, selection_year)
        prediction = _gated(folds[year], predictions[year], weight)
        selected[year] = prediction
        delta = brier_score(folds[year]["target"], prediction) - brier_score(
            folds[year]["target"], folds[year]["v2"]
        )
        selection_rows.append(
            {
                "season": year,
                "selection_year": selection_year or "no_prior",
                "weight": weight,
                "audit_delta": delta,
                "outer_target_used_for_selection": False,
            }
        )
    selection = pd.DataFrame(selection_rows)
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
        seed=20260819,
    )
    weighted_delta = float(
        np.average(selection["audit_delta"], weights=(1, 2, 3, 4))
    )
    (out_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (out_dir / "decision.md").write_text(
        "# Corrected-state CatBoost OOF\n\n"
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
                "recipe": "CB-R1 with corrected immediately-prior season state",
                "weights": WEIGHTS,
                "fit_rows": fit_rows,
                "fold_seconds": fold_seconds,
                "runtime_seconds": time.perf_counter() - started,
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(args.data_project, args.research_project, args.out_dir, args.n_resamples)


if __name__ == "__main__":
    main()
