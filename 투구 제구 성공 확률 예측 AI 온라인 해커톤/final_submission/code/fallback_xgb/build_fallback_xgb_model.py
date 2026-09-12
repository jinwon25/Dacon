"""Fit the fallback release-profile XGB on official training data.

This produces only a frozen train-derived asset.  It does not read `test.csv`,
write a submission, or modify the parent package.

The team 114-feature contract is reconstructed from frozen runtime assets.
--allow-reconstructed-training acknowledges that exact CUDA booster parity has not
yet been verified; the submitted inference asset is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
PROTOCOL = "FALLBACK_XGB_MODEL_FIT_V1"
ANCHOR_SEASON = 2024
HALF_LIFE_SEASONS = 2.0

PARAMS = dict(
    n_estimators=1800, learning_rate=.006, max_depth=10, min_child_weight=6000,
    subsample=.7, colsample_bytree=.5, reg_lambda=50., reg_alpha=1.,
    tree_method="hist", device="cuda:0", eval_metric="logloss", verbosity=0,
)
RANDOM_STATE = 2028


def verify_training_device(device: str) -> str:
    """Reject silent CUDA-to-CPU fallback before loading the training data."""
    if device == "cpu":
        return "cpu"
    if not (device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("Use cpu, cuda, or cuda:<device index>")
    probe = xgb.train(
        {"device": device, "tree_method": "hist", "objective": "binary:logistic",
         "max_depth": 1, "verbosity": 0},
        xgb.DMatrix(np.asarray([[0.], [1.], [2.], [3.]], dtype=np.float32),
                    label=np.asarray([0, 1, 0, 1], dtype=np.float32)),
        num_boost_round=1,
    )
    actual = json.loads(probe.save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith("cuda"):
        raise RuntimeError("CUDA requested but XGBoost selected CPU. Use a working GPU; "
                           "--device cpu is a separate, non-equivalent experiment.")
    return actual


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _runtime_module():
    spec = importlib.util.spec_from_file_location(
        "fallback_xgb_frozen_runtime", ROOT / "fallback_xgb_frozen_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT.parent / "data" / "train.csv")
    parser.add_argument("--asset-dir", type=Path, default=ROOT,
                        help="directory holding fallback_lookups.joblib and feature_columns.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=PARAMS["device"],
                        help='Training device; exact submitted-model parity is not verified on either device')
    parser.add_argument("--trackman-csv", type=Path, default=ROOT.parent / "data" / "trackman_history.csv")
    parser.add_argument("--allow-reconstructed-training", action="store_true")
    args = parser.parse_args()
    if not args.allow_reconstructed_training:
        parser.error("Exact CUDA training parity is unverified. "
                     "See 04_SUBMISSION_CHECKLIST.md. Reconstruction requires explicit opt-in.")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory; never replace submitted assets")
    if xgb.__version__ != "3.2.0":
        raise RuntimeError("Use XGBoost 3.2.0 in a separate training environment")
    actual_device = verify_training_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT.parent))
    from src.archive.v217_rebuild_fallback_xgb_oof import build_features

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    trackman = pd.read_csv(args.trackman_csv, encoding="utf-8-sig", low_memory=False)
    pitcher_map = pd.read_csv(ROOT / "pitcher_map.csv", encoding="utf-8-sig")
    columns = json.loads((args.asset_dir / "feature_columns.json").read_text(encoding="utf-8"))
    features = build_features(train, trackman, pitcher_map, columns)
    target = train.control_success.to_numpy(np.float32)
    season = train.season.to_numpy()
    weights = (.5 ** ((ANCHOR_SEASON - season) / HALF_LIFE_SEASONS)).astype(np.float32)

    params = dict(PARAMS, device=args.device)
    model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE)
    model.fit(features, target, sample_weight=weights)
    fitted_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if fitted_device != actual_device:
        raise RuntimeError("Training device changed after the preflight probe")
    model.save_model(args.output_dir / "fallback_xgb.json")

    (args.output_dir / "feature_columns.json").write_text(
        json.dumps(list(features.columns), ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "params": dict(params, random_state=RANDOM_STATE),
                "actual_training_device": fitted_device,
                "libraries": {name: importlib.metadata.version(name) for name in
                              ("xgboost", "numpy", "pandas", "scikit-learn")},
                "official_input_sha256": {
                    "train.csv": file_sha256(args.train_csv),
                    "trackman_history.csv": file_sha256(args.trackman_csv),
                },
                "exact_submitted_model_reproduction": False,
                "training_feature_source": "team 114-feature reconstruction; exact CUDA parity unverified",
                "train_rows": int(len(train)),
                "feature_count": int(features.shape[1]),
                "target": "control_success",
                "history_rule": "official_train_only",
                "test_rule": "row-local lookup only",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", args.output_dir, "features", features.shape[1], "rows", len(train))


if __name__ == "__main__":
    main()
