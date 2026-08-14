"""Train and persist the production corrected-state residual model.

The residual target is built from frozen 2021--2024 V2 nested OOF
predictions.  The production model therefore never sees an in-sample V2
prediction as its target.  Compact entity endpoints are also exported so
2025 inference can reproduce the corrected within-season state without
inspecting the evaluation batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.corrected_state_residual_oof import YEARS, _load_folds, _make_features
from src.data import read_main
from src.top1100_features import _prior_table


MODEL_NAME = "corrected_state_residual_lgb.txt"
SPEC_NAME = "corrected_state_residual_spec.json"
PITCHER_PRIOR_NAME = "corrected_state_pitcher_prior.csv"
BATTER_PRIOR_NAME = "corrected_state_batter_prior.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _last_endpoint(
    train: pd.DataFrame,
    entity: str,
    n_column: str,
    rate_column: str,
) -> pd.DataFrame:
    prior = _prior_table(train, entity, n_column, rate_column)
    prior = (
        prior.sort_values([entity, "__prior_season"], kind="mergesort")
        .drop_duplicates(entity, keep="last")
        .reset_index(drop=True)
    )
    if prior[entity].duplicated().any():
        raise ValueError(f"duplicate frozen endpoint for {entity}")
    return prior


def run(data_project: Path, research_project: Path, out_dir: Path) -> Path:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    train_path = data_project / "data" / "train.csv"
    train = read_main(train_path)
    folds = _load_folds(research_project, train)
    features = _make_features(train)

    fit_indices = np.concatenate(
        [folds[year]["train_index"].to_numpy(np.int64) for year in YEARS]
    )
    residual = np.concatenate(
        [
            folds[year]["target"].to_numpy(np.float64)
            - folds[year]["v2"].to_numpy(np.float64)
            for year in YEARS
        ]
    )
    if pd.Index(train.iloc[fit_indices]["row_id"].astype(str)).has_duplicates:
        raise ValueError("OOF rows overlap across outer seasons")

    fit_features = features.iloc[fit_indices].copy().reset_index(drop=True)
    categorical = list(
        fit_features.select_dtypes(include=["category", "object", "string"]).columns
    )
    vocabularies: dict[str, list[str]] = {}
    for column in categorical:
        values = fit_features[column].astype("string").fillna("__MISSING__")
        vocabulary = pd.Index(values.unique()).astype(str).tolist()
        vocabularies[column] = vocabulary
        fit_features[column] = pd.Categorical(values, categories=vocabulary)

    dataset = lgb.Dataset(
        fit_features,
        label=residual,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 2000,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 50.0,
            "verbosity": -1,
            "seed": 20260814,
            "num_threads": 6,
            "deterministic": True,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=160,
        callbacks=[lgb.log_evaluation(0)],
    )
    model_path = out_dir / MODEL_NAME
    model_path.write_text(model.model_to_string(), encoding="utf-8")

    pitcher_prior = _last_endpoint(
        train, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    batter_prior = _last_endpoint(
        train, "batter_id", "asof_batter_n", "asof_batter_success_rate"
    )
    pitcher_path = out_dir / PITCHER_PRIOR_NAME
    batter_path = out_dir / BATTER_PRIOR_NAME
    pitcher_prior.to_csv(pitcher_path, index=False, encoding="utf-8")
    batter_prior.to_csv(batter_path, index=False, encoding="utf-8")

    spec = {
        "version": 1,
        "method": "corrected_state_residual",
        "apply_game_type": "R",
        "eta": 1.0,
        "correction_clip": [-0.04, 0.04],
        "training_seasons": list(YEARS),
        "training_rows": int(len(fit_features)),
        "feature_columns": list(fit_features.columns),
        "categorical_columns": categorical,
        "categorical_vocabularies": vocabularies,
        "num_boost_round": 160,
        "seed": 20260814,
        "train_sha256": _sha256(train_path),
    }
    spec_path = out_dir / SPEC_NAME
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (model_path, spec_path, pitcher_path, batter_path)
        },
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "pitcher_endpoints": int(len(pitcher_prior)),
        "batter_endpoints": int(len(batter_prior)),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(out_dir), **manifest}, ensure_ascii=False, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_project, args.research_project, args.out_dir)


if __name__ == "__main__":
    main()
