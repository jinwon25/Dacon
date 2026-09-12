"""Fit deployable full-data command and batter experts for the v261 formula."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET


PROTOCOL = "V263_FINALIZE_CALENDAR_EXPERT_MODELS_V1"
DEPLOYMENT_YEAR = 2025
COMMAND_RANDOM_STATE = 2063
BATTER_RANDOM_STATE = 2064


def deployment_fit_mask(
    season: np.ndarray, is_futures: np.ndarray, year: int = DEPLOYMENT_YEAR
) -> np.ndarray:
    return (season < int(year)) & ~(is_futures & (season <= 2022))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _combine(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(supplement):
        raise ValueError("base and supplement row counts differ")
    if set(base).intersection(supplement):
        raise ValueError("base and supplement columns overlap")
    return pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)],
        axis=1,
        copy=False,
    )


def _fit_model(
    features: pd.DataFrame,
    target: np.ndarray,
    fit: np.ndarray,
    weight: np.ndarray,
    random_state: int,
    output_path: Path,
) -> float:
    started = time.time()
    model = xgb.XGBClassifier(**PARAMS, random_state=random_state)
    model.fit(features.loc[fit], target[fit], sample_weight=weight)
    model.save_model(output_path)
    del model
    gc.collect()
    return time.time() - started


def run(
    train_csv: Path,
    base_feature_cache: Path,
    command_feature_cache: Path,
    batter_feature_cache: Path,
    command_profile_cache: Path,
    batter_profile_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv, usecols=[TARGET, "season", "game_type"], low_memory=False
    )
    base = pd.read_parquet(base_feature_cache)
    command = pd.read_parquet(command_feature_cache)
    batter = pd.read_parquet(batter_feature_cache)
    if len(train) != len(base):
        raise ValueError("train/base row count mismatch")
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    fit = deployment_fit_mask(season, is_futures)
    weight = (
        0.5 ** ((DEPLOYMENT_YEAR - 1 - season[fit]) / 2.0)
    ).astype(np.float32)

    command_model = output_dir / "command_expert_xgb.json"
    batter_model = output_dir / "batter_expert_xgb.json"
    timing: dict[str, float] = {}
    if not command_model.exists():
        command_features = _combine(base, command)
        timing["command_fit_seconds"] = _fit_model(
            command_features,
            target,
            fit,
            weight,
            COMMAND_RANDOM_STATE,
            command_model,
        )
        (output_dir / "command_feature_columns.json").write_text(
            json.dumps(list(command_features.columns), ensure_ascii=False),
            encoding="utf-8",
        )
        del command_features
        gc.collect()
    else:
        timing["command_fit_seconds"] = 0.0
    if not batter_model.exists():
        batter_features = _combine(base, batter)
        timing["batter_fit_seconds"] = _fit_model(
            batter_features,
            target,
            fit,
            weight,
            BATTER_RANDOM_STATE,
            batter_model,
        )
        (output_dir / "batter_feature_columns.json").write_text(
            json.dumps(list(batter_features.columns), ensure_ascii=False),
            encoding="utf-8",
        )
        del batter_features
        gc.collect()
    else:
        timing["batter_fit_seconds"] = 0.0

    command_profiles = pd.read_parquet(command_profile_cache)
    batter_profiles = pd.read_parquet(batter_profile_cache)
    joblib.dump(command_profiles, output_dir / "command_profiles.joblib", compress=3)
    joblib.dump(batter_profiles, output_dir / "batter_profiles.joblib", compress=3)
    artifacts = [
        command_model,
        batter_model,
        output_dir / "command_feature_columns.json",
        output_dir / "batter_feature_columns.json",
        output_dir / "command_profiles.joblib",
        output_dir / "batter_profiles.joblib",
    ]
    summary = {
        "protocol": PROTOCOL,
        "status": "final_models_ready",
        "deployment_year": DEPLOYMENT_YEAR,
        "fit_rows": int(fit.sum()),
        "excluded_rows": int((~fit).sum()),
        "command_feature_count": int(base.shape[1] + command.shape[1]),
        "batter_feature_count": int(base.shape[1] + batter.shape[1]),
        "command_profile_rows": int(len(command_profiles)),
        "batter_profile_rows": int(len(batter_profiles)),
        "timing": timing,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifacts
        },
        "restrictions": {
            "official_train_only": True,
            "prior_season_target_free_trackman_profiles": True,
            "v217_hyperparameters_and_time_weights_frozen": True,
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
    parser.add_argument("--base-feature-cache", type=Path, required=True)
    parser.add_argument("--command-feature-cache", type=Path, required=True)
    parser.add_argument("--batter-feature-cache", type=Path, required=True)
    parser.add_argument("--command-profile-cache", type=Path, required=True)
    parser.add_argument("--batter-profile-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_feature_cache,
        args.command_feature_cache,
        args.batter_feature_cache,
        args.command_profile_cache,
        args.batter_profile_cache,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
