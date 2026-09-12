"""Fresh-fit the TrackMan LightGBM and frozen pitcher profiles used by the parent."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data import TARGET_COL, read_main, read_trackman
from src.features import FeatureBuilder
from src.trackman_linkage import build_pitcher_profile_table
from src.train import _base_lgb_params


EXPECTED = {
    "train.csv": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "trackman_history.csv": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
}
REQUIRED_VERSIONS = {
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "lightgbm": "4.6.0",
}
NUM_ITERATIONS = 41


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/trackman_linkage.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    config_path = args.config.resolve()
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
    for name, expected in EXPECTED.items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"Official data hash mismatch: {name}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    train = read_main(data_dir / "train.csv")
    trackman = read_trackman(data_dir / "trackman_history.csv")
    origins = range(int(train["season"].min()), int(train["season"].max()) + 2)
    profiles, _ = build_pitcher_profile_table(
        train, trackman, origins, config["linkage"]
    )
    output.mkdir(parents=True)
    profiles.loc[profiles["season"].eq(2025)].to_csv(
        output / "trackman_pitcher_profiles.csv", index=False, encoding="utf-8"
    )

    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    builder = FeatureBuilder(feature_set="trackman_pitcher", trackman_context=profiles)
    matrix = builder.fit_transform(train, target)
    dataset = lgb.Dataset(
        matrix,
        label=target,
        categorical_feature=builder.categorical_columns,
        free_raw_data=True,
    )
    booster = lgb.train(
        _base_lgb_params(config["model"]),
        dataset,
        num_boost_round=NUM_ITERATIONS,
        callbacks=[lgb.log_evaluation(0)],
    )
    (output / "trackman_lgb_model.txt").write_text(
        booster.model_to_string(), encoding="utf-8"
    )
    builder.save_spec(output / "trackman_feature_spec.json")
    report = {
        "protocol": "PARENT_TRACKMAN_FRESH_FIT_V1",
        "official_input_sha256": EXPECTED,
        "rows": int(len(train)),
        "profile_rows_all_origins": int(len(profiles)),
        "profile_rows_deployed_2025": int(profiles["season"].eq(2025).sum()),
        "feature_count": int(matrix.shape[1]),
        "model": config["model"],
        "iterations": NUM_ITERATIONS,
        "versions": versions,
        "files_sha256": {
            name: digest(output / name)
            for name in (
                "trackman_lgb_model.txt",
                "trackman_feature_spec.json",
                "trackman_pitcher_profiles.csv",
            )
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
