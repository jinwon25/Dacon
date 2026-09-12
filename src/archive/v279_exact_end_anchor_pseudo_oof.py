"""Pseudo-deployment XGB with genuinely current-season ASOF features.

The frozen fallback runtime historically subtracts the first ASOF state of the
last training season.  Because ASOF counts are career-cumulative, that makes a
future-season feature contain two seasons.  This experiment instead freezes
the last observable pre-pitch state in prior official-train history and uses
that same transform for every pseudo-deployment training and audit block.

No state is learned from the audit block.  The one unobserved final pitch in a
prior block is deliberately left in the delta; this keeps every anchor
constructible from the prior block alone and costs at most one count per
player.
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

from src.team_assets.JY_fallback_XGB_active50_w030 import fallback_xgb_frozen_runtime as runtime
from src.archive.v217_rebuild_fallback_xgb_oof import PARAMS, SPECS, TARGET, bss
from src.archive.v242_runtime_faithful_fallback_oof import build_frozen_lookup
from src.archive.v267_pseudo_deployment_xgb_oof import PSEUDO_START_YEAR, pseudo_fit_years


PROTOCOL = "V279_EXACT_END_ANCHOR_PSEUDO_OOF_V1"
RANDOM_STATE = 2079


def _pairs(
    frame: pd.DataFrame, key: str, values: list[str]
) -> dict[int, tuple[float, ...]]:
    return {
        int(index): tuple(float(value) for value in row)
        for index, row in frame.set_index(key)[values].iterrows()
    }


def build_exact_frozen_lookup(
    history: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
) -> dict[str, Any]:
    """Freeze each entity's latest observable career-cumulative state."""

    payload = build_frozen_lookup(history, trackman, pitcher_map)
    for id_column, n_column, rate_column, prefix in SPECS:
        values = history[[id_column, n_column, rate_column]].copy()
        values["succ"] = (
            pd.to_numeric(values[n_column], errors="coerce")
            * pd.to_numeric(values[rate_column], errors="coerce").fillna(0.0)
        )
        latest_index = values.groupby(id_column, sort=False)[n_column].idxmax()
        latest = values.loc[latest_index, [id_column, n_column, "succ"]]
        payload["anchors"][prefix] = _pairs(
            latest, id_column, [n_column, "succ"]
        )
    payload["features_version"] = 2
    payload["anchor_semantics"] = "latest_prior_pre_pitch_state"
    payload["multiscale_semantics"] = "same_exact_end_anchor"
    return payload


def build_exact_features(
    frame: pd.DataFrame,
    lookup: dict[str, Any],
    feature_columns: list[str],
    scratch_asset: Path,
) -> pd.DataFrame:
    """Reuse the frozen base transform, then correct every season feature."""

    scratch_asset.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookup, scratch_asset / "fallback_lookups.joblib", compress=3)
    (scratch_asset / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False), encoding="utf-8"
    )
    output = runtime.build(frame, scratch_asset)
    for id_column, n_column, rate_column, prefix in SPECS:
        ids = frame[id_column].astype(int).to_numpy()
        pairs = lookup["anchors"][prefix]
        anchor_n = np.asarray(
            [pairs.get(int(value), (0.0, 0.0))[0] for value in ids],
            dtype=np.float64,
        )
        anchor_sum = np.asarray(
            [pairs.get(int(value), (0.0, 0.0))[1] for value in ids],
            dtype=np.float64,
        )
        n = pd.to_numeric(frame[n_column], errors="coerce").to_numpy(np.float64)
        rate = pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(
            np.float64
        )
        prior = float(lookup["priors"][prefix])
        delta_n = np.maximum(np.nan_to_num(n) - anchor_n, 0.0)
        delta_sum = np.maximum(np.nan_to_num(n * rate) - anchor_sum, 0.0)
        season_rate = (delta_sum + 150.0 * prior) / (delta_n + 150.0)
        output[f"{prefix}_ssn"] = season_rate.astype(np.float32)
        output[f"{prefix}_ssn_vs_car"] = (
            season_rate - np.nan_to_num(rate, nan=prior)
        ).astype(np.float32)
        if prefix in ("p_succ", "b_succ"):
            output[f"{prefix}_ssn_n"] = delta_n.astype(np.float32)
            for shrink in (25, 75, 400, 1000):
                output[f"{prefix}_k{shrink}"] = (
                    (delta_sum + shrink * prior) / (delta_n + shrink)
                ).astype(np.float32)

    ppa = output["p_ppa"].to_numpy(np.float64)
    pitcher_n = output["p_succ_ssn_n"].to_numpy(np.float64)
    output["p_est_apps"] = (pitcher_n / np.clip(ppa, 5.0, None)).astype(
        np.float32
    )
    output["p_ssn_per_month"] = (
        pitcher_n
        / np.clip(frame["game_month"].to_numpy(np.float64), 3.0, None)
    ).astype(np.float32)
    return output.reindex(columns=feature_columns).astype(np.float32)


def _ensure_exact_block(
    year: int,
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    cache = output_dir / f"exact_runtime_features_{year}.parquet"
    if cache.exists():
        result = pd.read_parquet(cache)
    else:
        history = train.loc[train["season"].lt(year)].reset_index(drop=True)
        rows = train.loc[
            train["season"].eq(year)
            & train["game_type"].astype(str).eq("R")
        ].reset_index(drop=True)
        lookup = build_exact_frozen_lookup(history, trackman, pitcher_map)
        result = build_exact_features(
            rows,
            lookup,
            feature_columns,
            output_dir / f"lookup_{year}",
        )
        result.to_parquet(cache, index=False)
    if list(result.columns) != feature_columns:
        raise ValueError(f"feature-column mismatch: {year}")
    return result


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    feature_columns_json: Path,
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
        blocks[year] = _ensure_exact_block(
            year,
            train,
            trackman,
            pitcher_map,
            feature_columns,
            output_dir,
        )
        mask = (
            train["season"].eq(year)
            & train["game_type"].astype(str).eq("R")
        ).to_numpy()
        block_targets[year] = target[mask]
        if len(blocks[year]) != len(block_targets[year]):
            raise ValueError(f"target mismatch: {year}")

    folds: dict[str, Any] = {}
    for audit_year in audit_years:
        prediction_path = output_dir / f"exact_end_anchor_xgb_{audit_year}.npy"
        model_path = output_dir / f"exact_end_anchor_xgb_{audit_year}.json"
        fit_years = pseudo_fit_years(audit_year)
        if prediction_path.exists():
            prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
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
                f"[v279] audit={audit_year} fit_years={fit_years} "
                f"fit={sum(len(blocks[year]) for year in fit_years):,} "
                f"audit={len(blocks[audit_year]):,} "
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
        "status": "exact_end_anchor_pseudo_oof_built",
        "audit_years": list(audit_years),
        "feature_count": int(len(feature_columns)),
        "model_params": params,
        "folds": folds,
        "restrictions": {
            "official_train_only": True,
            "training_and_audit_share_exact_end_anchor_transform": True,
            "each_block_uses_strictly_prior_season_state": True,
            "one_unobserved_prior_final_pitch_left_in_delta": True,
            "regular_season_pseudo_blocks_only": True,
            "v217_tree_hyperparameters_frozen": True,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-years", type=int, nargs="+", default=[2024])
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.feature_columns_json,
        args.output_dir,
        tuple(args.audit_years),
        args.n_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
