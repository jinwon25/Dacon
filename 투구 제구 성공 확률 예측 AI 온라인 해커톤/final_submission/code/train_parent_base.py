"""Fresh-fit the base LightGBM and RandomForest used inside the submitted parent."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.data import ID_COL, TARGET_COL, read_main
from src.features import FeatureBuilder
from src.train import _base_lgb_params, make_official_rf


EXPECTED_TRAIN = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
REQUIRED_VERSIONS = {
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "lightgbm": "4.6.0",
    "scikit-learn": "1.6.1",
    "joblib": "1.5.1",
}
VARIANT = {
    "name": "lgb_engineered_l31",
    "feature_set": "engineered",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 500,
    "lambda_l2": 2.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
}
NUM_ITERATIONS = 54


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
    mismatch = {
        name: {"installed": versions[name], "required": required}
        for name, required in REQUIRED_VERSIONS.items()
        if versions[name] != required
    }
    if mismatch:
        raise RuntimeError("Version mismatch: " + json.dumps(mismatch))
    train_path = data_dir / "train.csv"
    if digest(train_path) != EXPECTED_TRAIN:
        raise ValueError("Official train.csv hash mismatch")

    output.mkdir(parents=True)
    train = read_main(train_path)
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    features = [column for column in train.columns if column not in (ID_COL, TARGET_COL)]

    builder = FeatureBuilder(feature_set="engineered")
    matrix = builder.fit_transform(train, target)
    dataset = lgb.Dataset(
        matrix,
        label=target,
        categorical_feature=builder.categorical_columns,
        free_raw_data=True,
    )
    booster = lgb.train(
        _base_lgb_params(VARIANT),
        dataset,
        num_boost_round=NUM_ITERATIONS,
        callbacks=[lgb.log_evaluation(0)],
    )
    (output / "lgb_model.txt").write_text(booster.model_to_string(), encoding="utf-8")
    builder.save_spec(output / "feature_spec.json")
    del matrix, dataset

    forest = make_official_rf(features)
    forest.fit(train[features], target)
    joblib.dump(forest, output / "rf_model.joblib", compress=3)

    report = {
        "protocol": "PARENT_BASE_FRESH_FIT_V1",
        "official_train_sha256": EXPECTED_TRAIN,
        "rows": int(len(train)),
        "feature_count": len(features),
        "lightgbm_variant": VARIANT,
        "lightgbm_iterations": NUM_ITERATIONS,
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 200,
            "random_state": 42,
        },
        "versions": versions,
        "files_sha256": {
            name: digest(output / name)
            for name in ("lgb_model.txt", "feature_spec.json", "rf_model.joblib")
        },
        "reference_model_read_during_fit": False,
        "test_values_read_during_fit": False,
        "private_score_recomputed": False,
    }
    (output / "training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
