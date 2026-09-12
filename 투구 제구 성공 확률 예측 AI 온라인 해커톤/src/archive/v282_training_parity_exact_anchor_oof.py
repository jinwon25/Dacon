"""Strict OOF for the original fallback learner with exact runtime anchors.

Unlike v279, which learns only from pseudo-deployment regular-season blocks,
this keeps the original v217 training recipe.  Training rows use their legal
within-season transform and audit rows use a frozen prior-end anchor that has
the same meaning at deployment.  Thus it isolates removal of the v242 runtime
feature mismatch while preserving the release learner and fit population.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import (
    PARAMS,
    TARGET,
    bss,
    build_features,
)
from src.archive.v279_exact_end_anchor_pseudo_oof import (
    build_exact_features,
    build_exact_frozen_lookup,
)


PROTOCOL = "V282_TRAINING_PARITY_EXACT_ANCHOR_OOF_V1"
RANDOM_STATE = 2028


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    feature_columns_json: Path,
    exact_feature_dir: Path,
    output_dir: Path,
    audit_years: tuple[int, ...],
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    trackman = pd.read_csv(trackman_csv, encoding="utf-8-sig", low_memory=False)
    pitcher_map = pd.read_csv(pitcher_map_csv)
    feature_columns = json.loads(feature_columns_json.read_text(encoding="utf-8"))
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    params = dict(PARAMS)
    params["n_jobs"] = int(n_jobs)
    folds: dict[str, Any] = {}

    for year in audit_years:
        prediction_path = output_dir / f"training_parity_exact_xgb_{year}.npy"
        model_path = output_dir / f"training_parity_exact_xgb_{year}.json"
        validation = (season == year) & ~is_futures
        fit = (season < year) & ~(is_futures & (season <= 2022))
        if prediction_path.exists():
            prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
            elapsed = 0.0
            reused = True
        else:
            started = time.time()
            history = train.loc[season < year].reset_index(drop=True)
            training_features = build_features(
                history, trackman, pitcher_map, feature_columns
            )
            training_mask = ~(
                history["game_type"].astype(str).eq("F").to_numpy()
                & history["season"].le(2022).to_numpy()
            )
            exact_cache = exact_feature_dir / f"exact_runtime_features_{year}.parquet"
            if exact_cache.exists():
                audit_features = pd.read_parquet(exact_cache)
            else:
                audit = train.loc[validation].reset_index(drop=True)
                lookup = build_exact_frozen_lookup(history, trackman, pitcher_map)
                audit_features = build_exact_features(
                    audit,
                    lookup,
                    feature_columns,
                    output_dir / f"lookup_{year}",
                )
                audit_features.to_parquet(exact_cache, index=False)
            if len(audit_features) != int(validation.sum()):
                raise ValueError(f"audit feature length mismatch: {year}")
            weight = (
                0.5
                ** (
                    (year - 1 - history.loc[training_mask, "season"].to_numpy())
                    / 2.0
                )
            ).astype(np.float32)
            model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE)
            model.fit(
                training_features.loc[training_mask],
                history.loc[training_mask, TARGET].to_numpy(np.float32),
                sample_weight=weight,
            )
            prediction = model.predict_proba(audit_features)[:, 1].astype(np.float64)
            model.save_model(model_path)
            np.save(prediction_path, prediction.astype(np.float32), allow_pickle=False)
            elapsed = time.time() - started
            reused = False
            del model, history, training_features, audit_features, weight
            gc.collect()
            print(
                f"[v282] year={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"bss={bss(target[validation], prediction):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "prediction": str(prediction_path),
            "model": str(model_path),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "training_parity_exact_anchor_oof_built",
        "audit_years": list(audit_years),
        "feature_count": int(len(feature_columns)),
        "model_params": params,
        "folds": folds,
        "restrictions": {
            "official_train_only": True,
            "labels_strictly_prior_to_audit_year": True,
            "training_uses_within_season_transform": True,
            "audit_uses_frozen_exact_prior_end_transform": True,
            "train_and_runtime_season_feature_meaning_matches": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--pitcher-map-csv", type=Path, required=True)
    parser.add_argument("--feature-columns-json", type=Path, required=True)
    parser.add_argument("--exact-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-years", type=int, nargs="+", default=[2024])
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.feature_columns_json,
        args.exact_feature_dir,
        args.output_dir,
        tuple(args.audit_years),
        args.n_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
