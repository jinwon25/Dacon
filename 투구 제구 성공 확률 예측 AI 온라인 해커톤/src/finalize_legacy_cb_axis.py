"""Train the frozen legacy CatBoost diversity axis for production.

This intentionally preserves the original CB-R1 feature recipe.  Its low
weight is selected on O23 and audited once on O24; it is not a replacement
for the corrected-state residual, only a partially independent add-on.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.data import read_main


MODEL_NAME = "legacy_cb_axis.cbm"
SPEC_NAME = "legacy_cb_axis_spec.json"
PITCHER_PRIOR_NAME = "legacy_cb_pitcher_prior.csv"
BATTER_PRIOR_NAME = "legacy_cb_batter_prior.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_legacy_features(research_project: Path):
    source = research_project / "src" / "top1100_features.py"
    module_spec = importlib.util.spec_from_file_location(
        "legacy_top1100_features", source
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not import legacy feature source: {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _last_endpoint(
    legacy_module,
    train: pd.DataFrame,
    entity: str,
    n_column: str,
    rate_column: str,
) -> pd.DataFrame:
    prior = legacy_module._prior_table(train, entity, n_column, rate_column)
    return (
        prior.sort_values([entity, "__prior_season"], kind="mergesort")
        .drop_duplicates(entity, keep="last")
        .reset_index(drop=True)
    )


def run(data_project: Path, research_project: Path, out_dir: Path) -> Path:
    data_project = data_project.resolve()
    research_project = research_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train_path = data_project / "data" / "train.csv"
    train = read_main(train_path)
    legacy = _load_legacy_features(research_project)

    features = legacy.build_features(train, train, include_ids=True)
    categorical = list(
        features.select_dtypes(include=["category", "object", "string"]).columns
    )
    feature_columns = list(features.columns)
    fit_index = np.arange(len(features), dtype=np.int64)
    if len(fit_index) > 1_200_000:
        rng = np.random.default_rng(42)
        fit_index = np.sort(rng.choice(fit_index, size=1_200_000, replace=False))
    fit = features.iloc[fit_index].reset_index(drop=True)
    target = train.iloc[fit_index]["control_success"].to_numpy(np.int8)
    del features
    gc.collect()
    for column in categorical:
        fit[column] = fit[column].astype("string").fillna("__MISSING__")

    model = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=30,
        random_strength=0.5,
        random_seed=42,
        thread_count=6,
        verbose=50,
        allow_writing_files=False,
        one_hot_max_size=32,
    )
    model.fit(fit, target, cat_features=categorical)
    model_path = out_dir / MODEL_NAME
    model.save_model(str(model_path))

    pitcher = _last_endpoint(
        legacy, train, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    batter = _last_endpoint(
        legacy, train, "batter_id", "asof_batter_n", "asof_batter_success_rate"
    )
    pitcher_path = out_dir / PITCHER_PRIOR_NAME
    batter_path = out_dir / BATTER_PRIOR_NAME
    pitcher.to_csv(pitcher_path, index=False, encoding="utf-8")
    batter.to_csv(batter_path, index=False, encoding="utf-8")

    spec = {
        "version": 1,
        "method": "legacy_cb_diversity_axis",
        "apply_game_type": "R",
        "effect_weight": 0.175,
        "selection": "O23 selected beta=0.35 x frozen CB gate=0.50",
        "o24_delta_with_corrected_residual": -0.0004908688130229044,
        "feature_columns": feature_columns,
        "categorical_columns": categorical,
        "training_rows": int(len(fit)),
        "iterations": 350,
        "seed": 42,
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
        "pitcher_endpoints": int(len(pitcher)),
        "batter_endpoints": int(len(batter)),
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
