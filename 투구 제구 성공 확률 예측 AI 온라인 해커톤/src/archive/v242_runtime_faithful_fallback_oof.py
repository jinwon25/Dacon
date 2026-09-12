"""Build fallback-XGB OOF with the actual frozen inference feature semantics.

The historical OOF builder generated audit-year features through the training
feature path.  The release, however, transforms a future season with lookups
frozen from all prior official-train rows and uses the zero-anchor multiscale
branch.  This module rebuilds every requested fold as a miniature final fit:
all transformation state is learned from season < audit_year, training rows
use the training transform, and audit rows use the frozen runtime transform.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.team_assets.JY_fallback_XGB_active50_w030 import fallback_xgb_frozen_runtime as runtime
from src.archive.v217_rebuild_fallback_xgb_oof import (
    PARAMS,
    SPECS,
    TARGET,
    TM_COLUMNS,
    _anchor_table,
    _appearance_table,
    _trackman_season_profiles,
    bss,
    build_features,
    situation_masks,
)


PROTOCOL = "V242_RUNTIME_FAITHFUL_FALLBACK_OOF_V1"


def _pairs(frame: pd.DataFrame, key: str, values: list[str]) -> dict[int, tuple[float, ...]]:
    return {
        int(index): tuple(float(value) for value in row)
        for index, row in frame.set_index(key)[values].iterrows()
    }


def build_frozen_lookup(
    history: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
) -> dict[str, Any]:
    """Create only-prior-season state consumed by the release runtime."""
    target = history[TARGET].to_numpy(np.float64)
    target_mean = float(target.mean())
    payload: dict[str, Any] = {
        "target_mean": target_mean,
        "priors": {},
        "cat": {},
        "anchors": {},
        "situations": {},
        "tm": {},
        "features_version": 1,
    }
    for column in runtime.CAT:
        payload["cat"][column] = {
            str(value): int(index)
            for index, value in enumerate(
                history[column].fillna("__NA__").astype(str).unique()
            )
        }

    for id_column, n_column, rate_column, prefix in SPECS:
        anchor = _anchor_table(history, id_column, n_column, rate_column)
        anchor = (
            anchor.sort_values("season")
            .groupby(id_column, sort=False)
            .tail(1)
        )
        payload["anchors"][prefix] = _pairs(
            anchor, id_column, [n_column, "succ"]
        )
        payload["priors"][prefix] = float(history[rate_column].mean())

    total = history.groupby("pitcher_id", sort=False)[TARGET].agg(["sum", "count"])
    raw_overall = total["sum"] / total["count"]
    overall = (total["sum"] + 300.0 * target_mean) / (total["count"] + 300.0)
    payload["overall"] = {int(key): float(value) for key, value in overall.items()}
    for name, selected in situation_masks(history).items():
        table = (
            history.loc[selected]
            .groupby("pitcher_id", sort=False)[TARGET]
            .agg(["sum", "count"])
            .reindex(total.index)
            .fillna(0.0)
        )
        rate = (table["sum"] + 300.0 * raw_overall) / (table["count"] + 300.0)
        payload["situations"][name] = {
            int(key): float(value) for key, value in rate.items()
        }

    pair_key = (
        history["pitcher_id"].astype("int64") * 100000
        + history["batter_id"].astype("int64")
    )
    pair = (
        history.assign(_key=pair_key)
        .groupby("_key", sort=False)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )
    payload["pb"] = _pairs(pair, "_key", ["sum", "count"])

    appearances = _appearance_table(history)
    ppa = appearances.groupby("pitcher_id", sort=False)["ppa"].median()
    payload["ppa"] = {int(key): float(value) for key, value in ppa.items()}
    payload["ppa_default"] = float(appearances["ppa"].median())

    profiles = _trackman_season_profiles(trackman, pitcher_map)
    profiles = profiles.loc[
        profiles["season"].le(int(history["season"].max()))
    ]
    profile = profiles.groupby("pitcher_id", sort=False)[list(TM_COLUMNS)].mean()
    for column in TM_COLUMNS:
        payload["tm"][column] = {
            int(key): float(value)
            for key, value in profile[column].dropna().items()
        }
    return payload


def build_runtime_audit_features(
    audit: pd.DataFrame,
    lookup: dict[str, Any],
    feature_columns: list[str],
    asset_dir: Path,
) -> pd.DataFrame:
    asset_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookup, asset_dir / "fallback_lookups.joblib", compress=3)
    (asset_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False), encoding="utf-8"
    )
    return runtime.build(audit, asset_dir)


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "all_fitted_state_strictly_prior_to_audit_year": True,
        "audit_uses_release_runtime_transform": True,
        "runtime_zero_anchor_multiscale_branch": True,
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
        checkpoint = output_dir / f"hyunku_runtime_faithful_xgb_{year}.npy"
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
        training_mask = ~(history_is_futures & (history["season"].to_numpy() <= 2022))
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
        model = xgb.XGBClassifier(**PARAMS, random_state=2028)
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
            f"[v242] year={year} fit={int(training_mask.sum()):,} "
            f"audit={int(validation.sum()):,} bss={folds[str(year)]['bss']:.6f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "protocol": PROTOCOL,
        "status": "runtime_faithful_oof_built",
        "years": list(years),
        "model_params": PARAMS,
        "folds": folds,
        "eligible_for_fallback_route_reaudit": len(folds) == len(years),
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
