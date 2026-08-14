from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output", default="submissions/lgbm_v1.csv")
    parser.add_argument(
        "--calibration-strength",
        type=float,
        default=1.0,
        help="0 disables validation calibration; 1 applies it fully.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    feature_columns = joblib.load(artifact_dir / "feature_columns.joblib")
    report = json.loads((artifact_dir / "training_report.json").read_text(encoding="utf-8"))

    print("Building test features...")
    X_all = build_features(args.data_dir, "test")
    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    indexed = sample.set_index(TIME_COL)

    if not indexed.index.equals(X_all.index):
        raise ValueError("Test feature timestamps do not exactly match sample_submission.csv")

    for target, capacity in CAPACITY_KWH.items():
        model = joblib.load(artifact_dir / f"{target}.joblib")
        columns = feature_columns[target] if isinstance(feature_columns, dict) else feature_columns
        X = X_all.reindex(columns=columns)
        settings = report["targets"][target]
        strength = args.calibration_strength
        scale = 1.0 + strength * (settings["scale"] - 1.0)
        offset = strength * settings["offset"]
        pred = model.predict(X) * scale + offset
        indexed[target] = np.clip(pred, 0, capacity)

    output = indexed.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(output):,} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
