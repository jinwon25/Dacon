"""Fit an independent LightGBM on v267 pseudo-deployment feature blocks."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.v217_rebuild_fallback_xgb_oof import TARGET, bss
from src.archive.v247_runtime_faithful_lightgbm_oof import PARAMS
from src.archive.v267_pseudo_deployment_xgb_oof import pseudo_fit_years


PROTOCOL = "V269_PSEUDO_DEPLOYMENT_LIGHTGBM_OOF_V1"


def run(
    train_csv: Path,
    pseudo_feature_dir: Path,
    output_dir: Path,
    audit_years: tuple[int, ...],
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv, usecols=[TARGET, "season", "game_type"], low_memory=False
    )
    target = train[TARGET].to_numpy(np.float32)
    params = dict(PARAMS)
    params["n_jobs"] = int(n_jobs)
    required_years = tuple(range(2021, max(int(y) for y in audit_years) + 1))
    blocks: dict[int, pd.DataFrame] = {}
    block_targets: dict[int, np.ndarray] = {}
    for year in required_years:
        path = pseudo_feature_dir / f"pseudo_runtime_features_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing v267 pseudo block: {path}")
        blocks[year] = pd.read_parquet(path)
        regular = train["season"].eq(year) & train["game_type"].astype(str).eq("R")
        block_targets[year] = target[regular.to_numpy()]
        if len(blocks[year]) != len(block_targets[year]):
            raise ValueError(f"pseudo block target mismatch: {year}")

    folds: dict[str, Any] = {}
    for audit_year in audit_years:
        prediction_path = output_dir / f"pseudo_deployment_lgbm_{audit_year}.npy"
        model_path = output_dir / f"pseudo_deployment_lgbm_{audit_year}.txt"
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
            model = lgb.LGBMClassifier(**params)
            model.fit(fit_features, fit_target, sample_weight=weight)
            prediction = model.predict_proba(blocks[audit_year])[:, 1].astype(np.float64)
            model.booster_.save_model(str(model_path))
            np.save(prediction_path, prediction.astype(np.float32), allow_pickle=False)
            elapsed = time.time() - started
            reused = False
            del model, fit_features, fit_target, fit_season, weight
            gc.collect()
            print(
                f"[v269] audit={audit_year} fit_years={fit_years} "
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
        "status": "pseudo_deployment_lightgbm_oof_built",
        "audit_years": list(audit_years),
        "feature_count": int(next(iter(blocks.values())).shape[1]),
        "model_params": params,
        "folds": folds,
        "restrictions": {
            "v267_pseudo_deployment_blocks_frozen": True,
            "v247_lightgbm_hyperparameters_frozen": True,
            "independent_learner_family": True,
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
    parser.add_argument("--pseudo-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.pseudo_feature_dir,
        args.output_dir,
        tuple(args.audit_years),
        args.n_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
