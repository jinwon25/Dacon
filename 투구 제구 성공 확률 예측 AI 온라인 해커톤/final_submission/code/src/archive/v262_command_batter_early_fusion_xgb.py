"""Strict-forward early fusion of pitcher command and batter exposure features.

Prior trials trained each TrackMan family independently and blended only their
probabilities.  This model concatenates the target-free prior-season pitcher
command-dispersion block and batter exposure block before tree construction so
the learner can represent pitcher-mechanics by batter-experience interactions.
The v217 training domain, weights, XGB capacity, and three audit years remain
frozen; only the joint representation changes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, TARGET, YEARS, bss


PROTOCOL = "V262_COMMAND_BATTER_EARLY_FUSION_XGB_STRICT_FORWARD_V1"
RANDOM_STATE = 2062
EXPECTED_BLOCK_WIDTHS = (114, 98, 89)


def combine_feature_blocks(
    base: pd.DataFrame,
    command: pd.DataFrame,
    batter: pd.DataFrame,
) -> pd.DataFrame:
    blocks = (base, command, batter)
    lengths = {len(block) for block in blocks}
    if len(lengths) != 1:
        raise ValueError("feature blocks have different row counts")
    widths = tuple(block.shape[1] for block in blocks)
    if widths != EXPECTED_BLOCK_WIDTHS:
        raise ValueError(
            f"feature block width mismatch: expected={EXPECTED_BLOCK_WIDTHS} got={widths}"
        )
    columns = [column for block in blocks for column in block.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("feature blocks contain duplicate columns")
    return pd.concat(
        [block.reset_index(drop=True) for block in blocks], axis=1, copy=False
    )


def fit_mask(season: np.ndarray, is_futures: np.ndarray, year: int) -> np.ndarray:
    return (season < int(year)) & ~(is_futures & (season <= 2022))


def run(
    train_csv: Path,
    base_feature_cache: Path,
    command_feature_cache: Path,
    batter_feature_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv, usecols=[TARGET, "season", "game_type"], low_memory=False
    )
    started = time.time()
    base = pd.read_parquet(base_feature_cache)
    command = pd.read_parquet(command_feature_cache)
    batter = pd.read_parquet(batter_feature_cache)
    features = combine_feature_blocks(base, command, batter)
    if len(features) != len(train):
        raise ValueError("feature/train row count mismatch")
    load_seconds = time.time() - started
    print(
        f"[v262] features={features.shape} load_seconds={load_seconds:.1f}",
        flush=True,
    )
    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"command_batter_fusion_xgb_{year}.npy"
        validation = (season == year) & ~is_futures
        fit = fit_mask(season, is_futures, year)
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            elapsed = 0.0
            reused = True
        else:
            weight = (0.5 ** ((year - 1 - season[fit]) / 2.0)).astype(np.float32)
            model = xgb.XGBClassifier(**PARAMS, random_state=RANDOM_STATE)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = model.predict_proba(features.loc[validation])[:, 1]
            elapsed = time.time() - started
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            reused = False
            print(
                f"[v262] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} "
                f"bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if len(prediction) != int(validation.sum()):
            raise ValueError(f"checkpoint length mismatch: {year}")
        folds[str(year)] = {
            "fit_rows": int(fit.sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "strict_forward_oof_fitted",
        "family_trial_count": 1,
        "feature_block_widths": list(EXPECTED_BLOCK_WIDTHS),
        "feature_count": int(features.shape[1]),
        "load_seconds": float(load_seconds),
        "folds": folds,
        "eligible_for_locked_route_audit": True,
        "eligible_for_packaging": False,
        "restrictions": {
            "v217_base_hyperparameters_domain_and_weights_frozen": True,
            "strict_forward_cached_feature_blocks_only": True,
            "pitcher_command_and_batter_exposure_early_fused": True,
            "single_preregistered_fusion_family": True,
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
    parser.add_argument("--base-feature-cache", type=Path, required=True)
    parser.add_argument("--command-feature-cache", type=Path, required=True)
    parser.add_argument("--batter-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_feature_cache,
        args.command_feature_cache,
        args.batter_feature_cache,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
