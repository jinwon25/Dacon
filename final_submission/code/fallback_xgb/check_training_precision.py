"""Rebuild training features and check their values against stored split boundaries.

Boundary membership is a necessary diagnostic for this histogram-trained model,
not proof of the full training matrix, row assignment, or retrained model parity.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from src.archive.v217_rebuild_fallback_xgb_oof import build_features


def check_boundaries(features, model):
    learner = model["learner"]
    if list(features.columns) != learner["feature_names"]:
        raise ValueError("Feature column order differs from stored model")
    cuts = {i: set() for i in range(features.shape[1])}
    for tree in learner["gradient_booster"]["model"]["trees"]:
        for node, left in enumerate(tree["left_children"]):
            if left != -1:
                cuts[tree["split_indices"][node]].add(tree["split_conditions"][node])
    details = []
    for index, column in enumerate(features.columns):
        boundaries = np.array(sorted(cuts[index]), dtype=np.float32)
        present = np.isin(boundaries, np.unique(features[column].dropna().to_numpy(dtype=np.float32)))
        details.append({"feature": column, "boundaries": len(boundaries),
                        "matched": int(present.sum()), "missing": boundaries[~present].tolist()})
    return {"rows": len(features), "features": features.shape[1],
            "boundaries": sum(r["boundaries"] for r in details),
            "matched": sum(r["matched"] for r in details),
            "all_boundaries_present": all(not r["missing"] for r in details),
            "exact_training_matrix_proven": False,
            "full_model_fresh_fit_proven": False, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new diagnostic output directory")
    print("Reading official training inputs", flush=True)
    train = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False)
    tm = pd.read_csv(args.trackman_csv, encoding="utf-8-sig", low_memory=False)
    mapping = pd.read_csv(ROOT / "pitcher_map.csv")
    columns = json.loads((ROOT / "feature_columns.json").read_text(encoding="utf-8"))
    print(f"Building {len(train)} x {len(columns)} training features", flush=True)
    features = build_features(train, tm, mapping, columns)
    report = check_boundaries(features, json.loads((ROOT / "fallback_xgb.json").read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "training_precision.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    if not report["all_boundaries_present"]:
        raise ValueError("Stored split boundaries disagree with training features")


if __name__ == "__main__":
    main()
