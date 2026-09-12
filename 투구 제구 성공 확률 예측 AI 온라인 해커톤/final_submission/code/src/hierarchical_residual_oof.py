"""Strict forward-OOF screen for frozen hierarchical residual lookups.

Every lookup is fitted from completed *training* OOF rows.  At inference a
row only performs a key lookup in that frozen table; no evaluation-batch
frequency, rank, lag, rolling value, or aggregate is computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.advanced_domain_residual_oof import _load_folds
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


PUBLIC_LEGACY_WEIGHT = 0.5534087425546688
ADVANCED_ETA = 0.90
SHRINKAGES = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
ETAS = (0.25, 0.50, 0.75, 1.00)

# Deliberately modest, baseball-interpretable frozen lookup candidates.
KEY_SPECS: dict[str, tuple[str, ...]] = {
    "pitcher": ("pitcher_id",),
    "pitcher_platoon": ("pitcher_id", "pitcher_hand", "batter_hand"),
    "pitcher_count": ("pitcher_id", "balls_before", "strikes_before"),
    "pitcher_month": ("pitcher_id", "game_month"),
    "pitcher_inning": ("pitcher_id", "inning_bucket"),
    "pitcher_base": ("pitcher_id", "base_state"),
    "batter": ("batter_id",),
    "batter_count": ("batter_id", "balls_before", "strikes_before"),
    "pitcher_batter": ("pitcher_id", "batter_id"),
    "pitcher_team": ("pitcher_team_id",),
    "batter_team": ("batter_team_id",),
    "count": ("balls_before", "strikes_before"),
    "platoon": ("pitcher_hand", "batter_hand"),
    "month": ("game_month",),
}


def _load_vector(path: Path, value: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as bundle:
        return bundle["row_id"].astype(str), bundle[value].astype(np.float64)


def _candidate_prediction(
    fold: pd.DataFrame,
    year: int,
    advanced_dir: Path,
    legacy_dirs: dict[int, Path],
) -> np.ndarray:
    advanced_id, correction = _load_vector(
        advanced_dir / f"correction_compact_l15_o{year}.npz", "correction"
    )
    legacy_id, legacy = _load_vector(
        legacy_dirs[year] / f"legacy_baseline_s42_o{year}.npz", "prediction"
    )
    expected = fold["row_id"].astype(str).to_numpy()
    if not np.array_equal(advanced_id, expected) or not np.array_equal(
        legacy_id, expected
    ):
        raise ValueError(f"OOF row mismatch for {year}")
    v2 = fold["v2"].to_numpy(np.float64)
    prediction = v2.copy()
    regular = fold["game_type"].eq("R").to_numpy()
    prediction[regular] = np.clip(
        v2[regular]
        + ADVANCED_ETA * correction[regular]
        + PUBLIC_LEGACY_WEIGHT * (legacy[regular] - v2[regular]),
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _fold_features(train: pd.DataFrame, fold: pd.DataFrame) -> pd.DataFrame:
    columns = sorted({column for keys in KEY_SPECS.values() for column in keys})
    raw_columns = [column for column in columns if column != "inning_bucket"]
    if "inning_bucket" in columns:
        raw_columns.append("inning")
    result = train.iloc[fold["train_index"].to_numpy()][raw_columns].reset_index(
        drop=True
    )
    result["inning_bucket"] = np.minimum(
        pd.to_numeric(result["inning"], errors="coerce").fillna(-1).astype("int16"),
        10,
    )
    return result.drop(columns=["inning"])


def _lookup_values(
    fit_features: pd.DataFrame,
    fit_residual: np.ndarray,
    audit_features: pd.DataFrame,
    keys: tuple[str, ...],
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    working = fit_features.loc[:, list(keys)].copy()
    working["_residual"] = fit_residual
    stats = (
        working.groupby(list(keys), observed=True, dropna=False)["_residual"]
        .agg(["sum", "count"])
        .reset_index()
    )
    stats["_correction"] = stats["sum"] / (stats["count"] + shrinkage)
    audit = audit_features.loc[:, list(keys)].copy()
    audit["_order"] = np.arange(len(audit), dtype=np.int64)
    mapped = audit.merge(stats, how="left", on=list(keys), sort=False).sort_values(
        "_order"
    )
    correction = mapped["_correction"].fillna(0.0).to_numpy(np.float64)
    count = mapped["count"].fillna(0.0).to_numpy(np.float64)
    return np.clip(correction, -0.05, 0.05), count


def run(
    data_project: Path,
    corrected_cb_dir: Path,
    advanced_dir: Path,
    legacy_o22_dir: Path,
    legacy_o23_dir: Path,
    legacy_o24_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(train, corrected_cb_dir)
    legacy_dirs = {
        2022: legacy_o22_dir,
        2023: legacy_o23_dir,
        2024: legacy_o24_dir,
    }
    frames = {year: _fold_features(train, folds[year]) for year in (2022, 2023, 2024)}
    base = {
        year: _candidate_prediction(folds[year], year, advanced_dir, legacy_dirs)
        for year in (2022, 2023, 2024)
    }
    target = {
        year: folds[year]["target"].to_numpy(np.float64)
        for year in (2022, 2023, 2024)
    }
    regular = {
        year: folds[year]["game_type"].eq("R").to_numpy()
        for year in (2022, 2023, 2024)
    }

    # Selection audit: fit O22 residuals, score untouched O23.
    fit22 = regular[2022]
    residual22 = (target[2022] - base[2022])[fit22]
    rows: list[dict[str, object]] = []
    for name, keys in KEY_SPECS.items():
        for shrinkage in SHRINKAGES:
            correction, count = _lookup_values(
                frames[2022].loc[fit22].reset_index(drop=True),
                residual22,
                frames[2023],
                keys,
                shrinkage,
            )
            for eta in ETAS:
                prediction = base[2023].copy()
                prediction[regular[2023]] = np.clip(
                    prediction[regular[2023]]
                    + eta * correction[regular[2023]],
                    1e-6,
                    1.0 - 1e-6,
                )
                score = brier_score(target[2023], prediction)
                rows.append(
                    {
                        "selection_year": 2023,
                        "name": name,
                        "keys": "+".join(keys),
                        "shrinkage": shrinkage,
                        "eta": eta,
                        "brier": score,
                        "delta_vs_base": score
                        - brier_score(target[2023], base[2023]),
                        "audit_nonzero_fraction": float(np.mean(count > 0)),
                        "audit_median_training_count": float(np.median(count[count > 0]))
                        if np.any(count > 0)
                        else 0.0,
                    }
                )
    grid = pd.DataFrame(rows).sort_values(["brier", "name", "shrinkage", "eta"])
    grid.to_csv(out_dir / "selection_o23_grid.csv", index=False)
    selected = grid.iloc[0].to_dict()

    # Locked forward audit: same recipe, now fit completed O22+O23 and score O24.
    name = str(selected["name"])
    keys = KEY_SPECS[name]
    shrinkage = float(selected["shrinkage"])
    eta = float(selected["eta"])
    fit_features = pd.concat(
        [frames[2022].loc[regular[2022]], frames[2023].loc[regular[2023]]],
        ignore_index=True,
    )
    fit_residual = np.concatenate(
        [
            (target[2022] - base[2022])[regular[2022]],
            (target[2023] - base[2023])[regular[2023]],
        ]
    )
    correction24, count24 = _lookup_values(
        fit_features, fit_residual, frames[2024], keys, shrinkage
    )
    prediction24 = base[2024].copy()
    prediction24[regular[2024]] = np.clip(
        prediction24[regular[2024]] + eta * correction24[regular[2024]],
        1e-6,
        1.0 - 1e-6,
    )
    delta24 = brier_score(target[2024], prediction24) - brier_score(
        target[2024], base[2024]
    )
    bootstrap = cluster_bootstrap_delta(
        target[2024],
        prediction24,
        base[2024],
        folds[2024]["pitcher_id"],
        n_resamples=5000,
        seed=20260814,
    )
    decision = {
        "base_definition": {
            "advanced_eta": ADVANCED_ETA,
            "legacy_weight": PUBLIC_LEGACY_WEIGHT,
        },
        "selected_on_2023": selected,
        "locked_2024_audit": {
            "base_brier": brier_score(target[2024], base[2024]),
            "candidate_brier": brier_score(target[2024], prediction24),
            "delta_vs_base": delta24,
            "nonzero_fraction": float(np.mean(count24 > 0)),
            "bootstrap": bootstrap,
        },
        "promote": bool(delta24 < 0.0 and bootstrap["ci_upper"] < 0.0),
        "training_rule": "completed official-train OOF rows only",
        "inference_rule": "current row key lookup in frozen training table only",
        "test_batch_aggregates_used": False,
    }
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "locked_o24_prediction.npz",
        row_id=folds[2024]["row_id"].astype(str).to_numpy(),
        prediction=prediction24,
        correction=correction24,
        training_count=count24,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--advanced-dir", type=Path, required=True)
    parser.add_argument("--legacy-o22-dir", type=Path, required=True)
    parser.add_argument("--legacy-o23-dir", type=Path, required=True)
    parser.add_argument("--legacy-o24-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(**vars(arguments))
