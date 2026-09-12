"""Fit the deployable v271 pseudo-XGB on 2021--2024 runtime blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET


PROTOCOL = "V276_FINALIZE_PSEUDO_DEPLOYMENT_XGB_V1"
DEPLOYMENT_YEAR = 2025
FIT_YEARS = (2021, 2022, 2023, 2024)
RANDOM_STATE = 2067


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(
    train_csv: Path,
    pseudo_feature_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv, usecols=[TARGET, "season", "game_type"], low_memory=False
    )
    target = train[TARGET].to_numpy(np.float32)
    blocks: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    seasons: list[np.ndarray] = []
    rows_by_year: dict[str, int] = {}
    for year in FIT_YEARS:
        path = pseudo_feature_dir / f"pseudo_runtime_features_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        block = pd.read_parquet(path)
        regular = train["season"].eq(year) & train["game_type"].astype(str).eq("R")
        block_target = target[regular.to_numpy()]
        if len(block) != len(block_target):
            raise ValueError(f"pseudo block target mismatch: {year}")
        blocks.append(block)
        targets.append(block_target)
        seasons.append(np.full(len(block), year, dtype=np.int16))
        rows_by_year[str(year)] = int(len(block))
    features = pd.concat(blocks, ignore_index=True, copy=False)
    fit_target = np.concatenate(targets)
    fit_season = np.concatenate(seasons)
    weight = (
        0.5 ** ((DEPLOYMENT_YEAR - 1 - fit_season) / 2.0)
    ).astype(np.float32)
    started = time.time()
    model = xgb.XGBClassifier(**PARAMS, random_state=RANDOM_STATE)
    model.fit(features, fit_target, sample_weight=weight)
    model_path = output_dir / "pseudo_deployment_xgb.json"
    columns_path = output_dir / "feature_columns.json"
    model.save_model(model_path)
    columns_path.write_text(
        json.dumps(list(features.columns), ensure_ascii=False), encoding="utf-8"
    )
    elapsed = time.time() - started
    summary = {
        "protocol": PROTOCOL,
        "status": "final_model_ready",
        "deployment_year": DEPLOYMENT_YEAR,
        "fit_years": list(FIT_YEARS),
        "fit_rows": int(len(features)),
        "rows_by_year": rows_by_year,
        "feature_count": int(features.shape[1]),
        "fit_seconds": float(elapsed),
        "model_params": PARAMS,
        "artifacts": {
            model_path.name: {
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
            },
            columns_path.name: {
                "bytes": columns_path.stat().st_size,
                "sha256": _sha256(columns_path),
            },
        },
        "restrictions": {
            "pseudo_runtime_blocks_only": True,
            "strictly_prior_state_per_training_block": True,
            "regular_season_only": True,
            "v217_tree_hyperparameters_frozen": True,
            "test_csv_read": False,
            "public_score_used_for_fit": False,
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
    parser.add_argument("--pseudo-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_csv, args.pseudo_feature_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
