"""Build an independent LightGBM OOF on the frozen fallback runtime features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.v217_rebuild_fallback_xgb_oof import (
    TARGET,
    bss,
    build_features,
)
from src.archive.v242_runtime_faithful_fallback_oof import (
    build_frozen_lookup,
    build_runtime_audit_features,
)


PROTOCOL = "V247_RUNTIME_FAITHFUL_LIGHTGBM_OOF_V1"
PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 1200,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 5000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.65,
    "reg_lambda": 50.0,
    "reg_alpha": 1.0,
    "max_bin": 63,
    "n_jobs": 16,
    "random_state": 2047,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "all_fitted_state_strictly_prior_to_audit_year": True,
        "audit_uses_release_runtime_transform": True,
        "same_feature_contract_as_deployed_fallback": True,
        "model_family_independent_from_xgboost": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    feature_columns_json: Path,
    output_dir: Path,
    years: tuple[int, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    trackman = pd.read_csv(trackman_csv, encoding="utf-8-sig", low_memory=False)
    pitcher_map = pd.read_csv(pitcher_map_csv)
    feature_columns = json.loads(feature_columns_json.read_text(encoding="utf-8"))
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, Any] = {}

    for year in years:
        checkpoint = output_dir / f"runtime_faithful_lgbm_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = (season < year) & ~(is_futures & (season <= 2022))
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            folds[str(year)] = {
                "fit_rows": int(fit.sum()),
                "audit_rows": int(validation.sum()),
                "bss": bss(target[validation], prediction),
                "checkpoint_reused": True,
            }
            continue

        started = time.time()
        history = train.loc[season < year].reset_index(drop=True)
        training_features = build_features(
            history, trackman, pitcher_map, feature_columns
        )
        history_is_futures = history["game_type"].astype(str).eq("F").to_numpy()
        training_mask = ~(
            history_is_futures & (history["season"].to_numpy() <= 2022)
        )
        lookup = build_frozen_lookup(history, trackman, pitcher_map)
        audit = train.loc[validation].reset_index(drop=True)
        audit_features = build_runtime_audit_features(
            audit,
            lookup,
            feature_columns,
            output_dir / f"lookup_{year}",
        )
        weight = (
            0.5 ** ((year - 1 - history.loc[training_mask, "season"].to_numpy()) / 2.0)
        ).astype(np.float32)
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            training_features.loc[training_mask],
            history.loc[training_mask, TARGET].to_numpy(np.float32),
            sample_weight=weight,
        )
        prediction = model.predict_proba(audit_features)[:, 1]
        np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
        elapsed = time.time() - started
        folds[str(year)] = {
            "fit_rows": int(training_mask.sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "checkpoint_reused": False,
            "audit_feature_count": int(audit_features.shape[1]),
        }
        print(
            f"[v247] year={year} fit={int(training_mask.sum()):,} "
            f"audit={int(validation.sum()):,} bss={folds[str(year)]['bss']:.6f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "protocol": PROTOCOL,
        "status": "runtime_faithful_lightgbm_oof_built",
        "years": list(years),
        "model_params": PARAMS,
        "folds": folds,
        "eligible_for_independent_route_audit": len(folds) == len(years),
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.feature_columns_json,
        args.output_dir,
        tuple(args.years),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
