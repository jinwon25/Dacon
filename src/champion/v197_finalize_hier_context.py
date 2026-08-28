"""Fit the official-train-only final hierarchy model for 2025 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.archive.v184_hierarchical_residual_stack import (
    MODEL_CONFIG,
    hierarchical_pitcher_features,
    prepare_model_frame,
)
from src.champion.v131_hoo_h1_independent_oof import _prepare_features
from src.champion.v197_hier_context_runtime import hierarchical_features_for_inference


PROTOCOL = "V197_FINAL_HIER_CONTEXT_MODEL_V1"


def build_opening_snapshot(
    frame: pd.DataFrame,
    target: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(history, dtype=bool)
    if not selected.any():
        raise ValueError("opening snapshot history is empty")
    pitcher = pd.to_numeric(frame.loc[selected, "pitcher_id"], errors="raise").to_numpy(np.int64)
    career_n = pd.to_numeric(
        frame.loc[selected, "asof_pitcher_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    rate = pd.to_numeric(
        frame.loc[selected, "asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    positions = np.flatnonzero(selected)
    rows = pd.DataFrame(
        {
            "pitcher_id": pitcher,
            "position": positions,
            "n_after": career_n + 1.0,
            "events_after": career_n * rate + np.asarray(target, dtype=np.float64)[selected],
        }
    )
    tail = rows.groupby("pitcher_id", sort=False).tail(1)
    opening = {
        int(row.pitcher_id): (float(row.n_after), float(row.events_after))
        for row in tail.itertuples(index=False)
    }
    return {
        "pitcher_opening": opening,
        "league_prior": float(np.asarray(target, dtype=np.float64)[selected].mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(
    train_csv: Path,
    trackman_csv: Path,
    external_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    hierarchical = hierarchical_pitcher_features(train, target, season)
    model_frame, feature_names, categorical = prepare_model_frame(
        train, h1_features, hierarchical
    )

    # Verify the frozen snapshot path reproduces the 2024 hierarchy exactly.
    snapshot23 = build_opening_snapshot(train, target, season < 2024)
    inferred24 = hierarchical_features_for_inference(
        train.loc[season == 2024].reset_index(drop=True), snapshot23
    )
    expected24 = hierarchical.loc[season == 2024].reset_index(drop=True)
    parity = float(np.nanmax(np.abs(
        inferred24.to_numpy(np.float64) - expected24.to_numpy(np.float64)
    )))
    if parity > 2e-6:
        raise ValueError(f"hierarchical inference parity failed: {parity}")

    weights = np.power(
        float(MODEL_CONFIG["decay"]), np.maximum(2024 - season, 0)
    )
    residual = target - hierarchical["hier_prediction"].to_numpy(np.float64)
    model = CatBoostRegressor(
        iterations=int(MODEL_CONFIG["iterations"]),
        learning_rate=float(MODEL_CONFIG["learning_rate"]),
        depth=int(MODEL_CONFIG["depth"]),
        l2_leaf_reg=float(MODEL_CONFIG["l2_leaf_reg"]),
        random_strength=float(MODEL_CONFIG["random_strength"]),
        bootstrap_type="Bernoulli",
        subsample=float(MODEL_CONFIG["subsample"]),
        loss_function="RMSE",
        random_seed=int(MODEL_CONFIG["seed"]),
        thread_count=16,
        verbose=50,
        allow_writing_files=False,
    )
    pool = Pool(
        model_frame[feature_names], label=residual, weight=weights,
        cat_features=categorical,
    )
    started = time.time()
    model.fit(pool)
    model_path = output_dir / "hier_context.cbm"
    model.save_model(model_path)

    snapshot = build_opening_snapshot(train, target, season <= 2024)
    snapshot.update(
        {
            "protocol": PROTOCOL,
            "target_year": 2025,
            "h1_features": list(h1_features),
            "feature_names": list(feature_names),
            "categorical": list(categorical),
            "model_config": dict(MODEL_CONFIG),
        }
    )
    snapshot_path = output_dir / "snapshot.joblib"
    joblib.dump(snapshot, snapshot_path, compress=3)
    summary = {
        "protocol": PROTOCOL,
        "status": "final_model_ready",
        "rows": int(len(train)),
        "seasons": sorted(int(value) for value in np.unique(season)),
        "feature_count": len(feature_names),
        "categorical_count": len(categorical),
        "pitcher_snapshot_count": len(snapshot["pitcher_opening"]),
        "league_prior": snapshot["league_prior"],
        "hierarchical_2024_parity_max_abs": parity,
        "fit_seconds": time.time() - started,
        "model": {"path": str(model_path), "bytes": model_path.stat().st_size, "sha256": _sha256(model_path)},
        "snapshot": {"path": str(snapshot_path), "bytes": snapshot_path.stat().st_size, "sha256": _sha256(snapshot_path)},
        "restrictions": {
            "official_train_only": True,
            "prior_season_official_trackman_only": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "other_test_rows_required": False,
            "public_score_used_for_training": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.external_root, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
