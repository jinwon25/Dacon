"""Fresh-fit the half-life-one RandomForest used by the parent hybrid."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib

from src.data import ID_COL, TARGET_COL, read_main
from src.recency_training import season_decay_weights
from src.train import make_official_rf


EXPECTED_TRAIN = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
REQUIRED_VERSIONS = {
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "joblib": "1.5.1",
}


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

    train = read_main(train_path)
    features = [column for column in train.columns if column not in (ID_COL, TARGET_COL)]
    weights = season_decay_weights(train["season"].to_numpy(), half_life=1.0)
    model = make_official_rf(features, random_state=42)
    model.fit(
        train[features],
        train[TARGET_COL],
        clf__sample_weight=weights,
    )
    output.mkdir(parents=True)
    path = output / "rf_recency_h1.joblib"
    joblib.dump(model, path, compress=3)
    report = {
        "protocol": "PARENT_RECENCY_RF_FRESH_FIT_V1",
        "official_train_sha256": EXPECTED_TRAIN,
        "rows": int(len(train)),
        "feature_count": len(features),
        "half_life_seasons": 1.0,
        "anchor_season": int(train["season"].max()),
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 200,
            "random_state": 42,
        },
        "versions": versions,
        "file_sha256": digest(path),
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
