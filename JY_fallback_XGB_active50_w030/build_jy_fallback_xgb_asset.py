"""Fit the independently validated fallback-TrackMan XGB on official training data.

This creates only a frozen train-derived asset.  It does not inspect test.csv,
write a submission, or alter the immutable JY champion package.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT.parents[1] / "reference-hyunku-lga-data" / "lga_data-main"
OUT = ROOT / "artifacts" / "private_oof_runtime" / "jy_fallback_xgb_asset"
sys.path.insert(0, str(HY))
import codex_reproduction.repro_v10a_core as rr

PARAMS = dict(
    n_estimators=1800, learning_rate=.006, max_depth=10, min_child_weight=6000,
    subsample=.7, colsample_bytree=.5, reg_lambda=50., reg_alpha=1.,
    tree_method="hist", device="cuda:0", eval_metric="logloss", verbosity=0,
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(HY / "data" / "train.csv", encoding="utf-8-sig")
    old = rr.ART
    rr.ART = OUT
    try:
        features = rr.add_public_context_features(train, train, rr.make_base_features(train, train))
    finally:
        rr.ART = old
    y = train.control_success.to_numpy(np.float32)
    season = train.season.to_numpy()
    weights = (.5 ** ((2024 - season) / 2.)).astype(np.float32)
    model = xgb.XGBClassifier(**PARAMS, random_state=2028)
    model.fit(features, y, sample_weight=weights)
    model.save_model(OUT / "fallback_xgb.json")
    (OUT / "feature_columns.json").write_text(json.dumps(list(features.columns), ensure_ascii=False), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps({"params": PARAMS, "train_rows": int(len(train)), "feature_count": int(features.shape[1]), "target": "control_success", "history_rule": "official_train_only", "test_rule": "row-local lookup only"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, "features", features.shape[1], "rows", len(train))


if __name__ == "__main__":
    main()
