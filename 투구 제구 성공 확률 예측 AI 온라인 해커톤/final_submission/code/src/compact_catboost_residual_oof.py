"""Compact CatBoost residual challenger with strict forward OOF training."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.advanced_domain_features import (
    add_base_prediction_features,
    build_advanced_domain_features,
)
from src.advanced_domain_residual_oof import _feature_columns, _load_folds
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


PUBLIC_WEIGHT = 0.5534087425546688
ETAS = (0.0, 0.25, 0.50, 0.65, 0.75, 0.90, 1.0, 1.10, 1.25)


def _clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    categorical = list(
        frame.select_dtypes(include=["category", "object", "string"]).columns
    )
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame, categorical


def _load_legacy(path: Path, fold: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        prediction = bundle["prediction"].astype(np.float64)
    if not np.array_equal(row_id, fold["row_id"].astype(str).to_numpy()):
        raise ValueError(f"legacy row mismatch: {path}")
    return prediction


def _prediction(
    fold: pd.DataFrame,
    correction: np.ndarray,
    legacy: np.ndarray,
    eta: float,
) -> np.ndarray:
    v2 = fold["v2"].to_numpy(np.float64)
    regular = fold["game_type"].eq("R").to_numpy()
    result = v2.copy()
    result[regular] = np.clip(
        v2[regular]
        + eta * correction[regular]
        + PUBLIC_WEIGHT * (legacy[regular] - v2[regular]),
        1e-6,
        1.0 - 1e-6,
    )
    return result


def run(
    data_project: Path,
    corrected_cb_dir: Path,
    legacy_o23: Path,
    legacy_o24: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project.resolve() / "data" / "train.csv")
    folds = _load_folds(train, corrected_cb_dir.resolve())
    domain = build_advanced_domain_features(train, train, include_teams=False)
    legacy = {
        2023: _load_legacy(legacy_o23.resolve(), folds[2023]),
        2024: _load_legacy(legacy_o24.resolve(), folds[2024]),
    }
    corrections: dict[int, np.ndarray] = {}
    fit_rows: dict[int, int] = {}
    fit_seconds: dict[int, float] = {}

    for audit_year in (2023, 2024):
        fit_years = tuple(year for year in (2021, 2022, 2023) if year < audit_year)
        fit_index = np.concatenate(
            [folds[year]["train_index"].to_numpy(np.int64) for year in fit_years]
        )
        fit_probability = np.concatenate(
            [folds[year]["v2"].to_numpy(np.float64) for year in fit_years]
        )
        residual = np.concatenate(
            [
                folds[year]["target"].to_numpy(np.float64)
                - folds[year]["v2"].to_numpy(np.float64)
                for year in fit_years
            ]
        )
        audit_index = folds[audit_year]["train_index"].to_numpy(np.int64)
        fit = add_base_prediction_features(
            domain.iloc[fit_index].reset_index(drop=True), fit_probability
        )
        audit = add_base_prediction_features(
            domain.iloc[audit_index].reset_index(drop=True),
            folds[audit_year]["v2"].to_numpy(np.float64),
        )
        columns = _feature_columns(fit, "compact")
        fit, categorical = _clean(fit[columns])
        audit, _ = _clean(audit[columns])
        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.035,
            loss_function="RMSE",
            l2_leaf_reg=50,
            random_strength=0.3,
            random_seed=20260822,
            thread_count=12,
            verbose=100,
            allow_writing_files=False,
            one_hot_max_size=32,
        )
        started = time.perf_counter()
        model.fit(fit, residual, cat_features=categorical)
        correction = np.clip(
            np.asarray(model.predict(audit), dtype=np.float64), -0.05, 0.05
        )
        seconds = time.perf_counter() - started
        corrections[audit_year] = correction
        fit_rows[audit_year] = len(fit)
        fit_seconds[audit_year] = seconds
        model.save_model(str(out_dir / f"compact_cb_residual_o{audit_year}.cbm"))
        np.savez_compressed(
            out_dir / f"compact_cb_residual_o{audit_year}.npz",
            row_id=folds[audit_year]["row_id"].astype(str).to_numpy(),
            correction=correction,
        )
        print(
            f"audit={audit_year} rows={len(fit)} seconds={seconds:.1f}", flush=True
        )
        del fit, audit, model
        gc.collect()

    rows = []
    predictions: dict[tuple[int, float], np.ndarray] = {}
    for year in (2023, 2024):
        target = folds[year]["target"].to_numpy(np.float64)
        v2 = folds[year]["v2"].to_numpy(np.float64)
        for eta in ETAS:
            prediction = _prediction(folds[year], corrections[year], legacy[year], eta)
            predictions[(year, eta)] = prediction
            rows.append(
                {
                    "season": year,
                    "eta": eta,
                    "brier": brier_score(target, prediction),
                    "delta_vs_v2": brier_score(target, prediction)
                    - brier_score(target, v2),
                    "fit_rows": fit_rows[year],
                    "fit_seconds": fit_seconds[year],
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)
    choice = metrics.loc[metrics["season"].eq(2023)].sort_values(
        ["brier", "eta"], kind="mergesort"
    ).iloc[0]
    eta = float(choice["eta"])
    audit = metrics.loc[
        metrics["season"].eq(2024) & metrics["eta"].eq(eta)
    ].iloc[0]
    lgb_reference_path = (
        out_dir.parent
        / "advanced_domain_residual_20260814_01"
        / "correction_compact_l15_o2024.npz"
    )
    with np.load(lgb_reference_path, allow_pickle=True) as bundle:
        lgb_correction = bundle["correction"].astype(np.float64)
    lgb_prediction = _prediction(
        folds[2024], lgb_correction, legacy[2024], 0.90
    )
    selected_prediction = predictions[(2024, eta)]
    bootstrap = cluster_bootstrap_delta(
        folds[2024]["target"].to_numpy(),
        selected_prediction,
        lgb_prediction,
        folds[2024]["pitcher_id"].astype(str).to_numpy(),
        n_resamples=n_resamples,
        seed=20260822,
    )
    decision = {
        "selection_season": 2023,
        "selected_eta": eta,
        "selection_brier": float(choice["brier"]),
        "audit_season": 2024,
        "audit_brier": float(audit["brier"]),
        "audit_delta_vs_lgb_compact_eta_0.90": brier_score(
            folds[2024]["target"], selected_prediction
        )
        - brier_score(folds[2024]["target"], lgb_prediction),
        "bootstrap_vs_lgb_compact": bootstrap,
        "outer_target_used_for_selection": False,
    }
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": "CatBoostRegressor compact residual",
                "iterations": 300,
                "depth": 5,
                "public_legacy_weight": PUBLIC_WEIGHT,
                "etas": ETAS,
                "test_batch_aggregates_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metrics.sort_values(["season", "brier"]).to_string(index=False))
    print(json.dumps(decision, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--legacy-o23", type=Path, required=True)
    parser.add_argument("--legacy-o24", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.data_project,
        args.corrected_cb_dir,
        args.legacy_o23,
        args.legacy_o24,
        args.out_dir,
        args.n_resamples,
    )


if __name__ == "__main__":
    main()
