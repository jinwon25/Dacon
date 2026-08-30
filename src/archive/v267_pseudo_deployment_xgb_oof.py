"""Train XGB only on historical rows transformed as future deployments.

Each regular-season block is rebuilt with state frozen at the end of the
previous season.  Models then learn from pooled pseudo-deployment blocks and
predict the next block with the same transform semantics.  This removes the
training-transform/runtime-transform shift documented by v242 without tuning
the v217 tree hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from JY_fallback_XGB_active50_w030 import fallback_xgb_frozen_runtime as runtime
from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET, bss
from src.archive.v242_runtime_faithful_fallback_oof import build_frozen_lookup


PROTOCOL = "V267_PSEUDO_DEPLOYMENT_XGB_OOF_V1"
PSEUDO_START_YEAR = 2021
RANDOM_STATE = 2067


def pseudo_fit_years(audit_year: int) -> tuple[int, ...]:
    if int(audit_year) <= PSEUDO_START_YEAR:
        raise ValueError("audit year must follow the pseudo start year")
    return tuple(range(PSEUDO_START_YEAR, int(audit_year)))


def _ensure_lookup(
    year: int,
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
    feature_columns: list[str],
    existing_lookup_dir: Path,
    output_dir: Path,
) -> Path:
    existing = existing_lookup_dir / f"lookup_{year}"
    if (existing / "fallback_lookups.joblib").exists():
        return existing
    target = output_dir / f"lookup_{year}"
    target.mkdir(parents=True, exist_ok=True)
    lookup_file = target / "fallback_lookups.joblib"
    if not lookup_file.exists():
        history = train.loc[train["season"].lt(year)].reset_index(drop=True)
        if history.empty:
            raise ValueError(f"no prior history for pseudo year {year}")
        lookup = build_frozen_lookup(history, trackman, pitcher_map)
        joblib.dump(lookup, lookup_file, compress=3)
        (target / "feature_columns.json").write_text(
            json.dumps(feature_columns, ensure_ascii=False), encoding="utf-8"
        )
    return target


def _build_pseudo_block(
    year: int,
    train: pd.DataFrame,
    lookup: Path,
    feature_columns: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    cache = output_dir / f"pseudo_runtime_features_{year}.parquet"
    if cache.exists():
        features = pd.read_parquet(cache)
    else:
        regular = train["season"].eq(year) & train["game_type"].astype(str).eq("R")
        rows = train.loc[regular].reset_index(drop=True)
        features = runtime.build(rows, lookup)
        features.to_parquet(cache, index=False)
    if list(features.columns) != feature_columns:
        raise ValueError(f"pseudo feature-column mismatch: {year}")
    return features


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    feature_columns_json: Path,
    existing_lookup_dir: Path,
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
    params = dict(PARAMS)
    params["n_jobs"] = int(n_jobs)

    required_years = tuple(
        range(PSEUDO_START_YEAR, max(int(year) for year in audit_years) + 1)
    )
    blocks: dict[int, pd.DataFrame] = {}
    block_targets: dict[int, np.ndarray] = {}
    for year in required_years:
        lookup = _ensure_lookup(
            year,
            train,
            trackman,
            pitcher_map,
            feature_columns,
            existing_lookup_dir,
            output_dir,
        )
        blocks[year] = _build_pseudo_block(
            year, train, lookup, feature_columns, output_dir
        )
        regular = train["season"].eq(year) & train["game_type"].astype(str).eq("R")
        block_targets[year] = target[regular.to_numpy()]
        if len(blocks[year]) != len(block_targets[year]):
            raise ValueError(f"pseudo block target mismatch: {year}")

    folds: dict[str, Any] = {}
    for audit_year in audit_years:
        prediction_path = output_dir / f"pseudo_deployment_xgb_{audit_year}.npy"
        model_path = output_dir / f"pseudo_deployment_xgb_{audit_year}.json"
        fit_years = pseudo_fit_years(audit_year)
        if prediction_path.exists():
            prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
            if len(prediction) != len(block_targets[audit_year]):
                raise ValueError(f"checkpoint length mismatch: {audit_year}")
            elapsed = 0.0
            reused = True
        else:
            started = time.time()
            fit_features = pd.concat(
                [blocks[year] for year in fit_years], ignore_index=True, copy=False
            )
            fit_target = np.concatenate([block_targets[year] for year in fit_years])
            fit_season = np.concatenate(
                [np.full(len(blocks[year]), year, dtype=np.int16) for year in fit_years]
            )
            weight = (
                0.5 ** ((audit_year - 1 - fit_season) / 2.0)
            ).astype(np.float32)
            model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE)
            model.fit(fit_features, fit_target, sample_weight=weight)
            prediction = model.predict_proba(blocks[audit_year])[:, 1].astype(np.float64)
            model.save_model(model_path)
            np.save(prediction_path, prediction.astype(np.float32), allow_pickle=False)
            elapsed = time.time() - started
            reused = False
            del model, fit_features, fit_target, fit_season, weight
            gc.collect()
            print(
                f"[v267] audit={audit_year} fit_years={fit_years} "
                f"fit={sum(len(blocks[year]) for year in fit_years):,} "
                f"audit_rows={len(blocks[audit_year]):,} "
                f"bss={bss(block_targets[audit_year], prediction):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(audit_year)] = {
            "fit_years": list(fit_years),
            "fit_rows": int(sum(len(blocks[year]) for year in fit_years)),
            "audit_rows": int(len(blocks[audit_year])),
            "bss": bss(block_targets[audit_year], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "prediction": str(prediction_path),
            "model": str(model_path),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "pseudo_deployment_oof_built",
        "pseudo_start_year": PSEUDO_START_YEAR,
        "audit_years": list(audit_years),
        "feature_count": int(len(feature_columns)),
        "model_params": params,
        "folds": folds,
        "restrictions": {
            "training_and_audit_share_release_runtime_transform": True,
            "each_block_uses_strictly_prior_season_state": True,
            "regular_season_pseudo_blocks_only": True,
            "v217_tree_hyperparameters_frozen": True,
            "test_csv_read": False,
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
    parser.add_argument("--existing-lookup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.feature_columns_json,
        args.existing_lookup_dir,
        args.output_dir,
        tuple(args.audit_years),
        args.n_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
