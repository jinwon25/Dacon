"""Train the production compact advanced-domain residual model.

Targets are based solely on frozen nested OOF predictions from official
training seasons 2021--2024.  The resulting model consumes row-local inputs
and frozen official-train entity endpoints at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.advanced_domain_features import (
    add_base_prediction_features,
    build_advanced_domain_features,
)
from src.advanced_domain_residual_oof import YEARS, _feature_columns, _load_folds
from src.data import read_main


MODEL_NAME = "advanced_domain_residual_lgb.txt"
SPEC_NAME = "advanced_domain_residual_spec.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(
    data_project: Path,
    corrected_cb_dir: Path,
    out_dir: Path,
) -> Path:
    data_project = data_project.resolve()
    corrected_cb_dir = corrected_cb_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    train_path = data_project / "data" / "train.csv"
    train = read_main(train_path)
    folds = _load_folds(train, corrected_cb_dir)
    domain = build_advanced_domain_features(train, train, include_teams=True)
    fit_indices = np.concatenate(
        [folds[year]["train_index"].to_numpy(np.int64) for year in YEARS]
    )
    base_probability = np.concatenate(
        [folds[year]["v2"].to_numpy(np.float64) for year in YEARS]
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

    fit = add_base_prediction_features(
        domain.iloc[fit_indices].reset_index(drop=True), base_probability
    )
    columns = _feature_columns(fit, "compact")
    fit = fit[columns].copy()
    categorical = list(
        fit.select_dtypes(include=["category", "object", "string"]).columns
    )
    vocabularies: dict[str, list[str]] = {}
    for column in categorical:
        values = fit[column].astype("string").fillna("__MISSING__")
        vocabulary = pd.Index(values.unique()).astype(str).tolist()
        vocabularies[column] = vocabulary
        fit[column] = pd.Categorical(values, categories=vocabulary)

    dataset = lgb.Dataset(
        fit,
        label=residual,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.025,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 2000,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 75.0,
            "verbosity": -1,
            "seed": 20260821,
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
    spec = {
        "version": 1,
        "method": "compact_advanced_domain_residual",
        "apply_game_type": "R",
        "eta": 0.90,
        "correction_clip": [-0.05, 0.05],
        "training_seasons": list(YEARS),
        "training_rows": int(len(fit)),
        "feature_columns": list(fit.columns),
        "categorical_columns": categorical,
        "categorical_vocabularies": vocabularies,
        "num_boost_round": 160,
        "seed": 20260821,
        "train_sha256": _sha256(train_path),
        "test_batch_aggregates_used": False,
    }
    spec_path = out_dir / SPEC_NAME
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (model_path, spec_path)
        },
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "feature_count": len(fit.columns),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(out_dir), **manifest}, ensure_ascii=False, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(**vars(arguments))


if __name__ == "__main__":
    main()
