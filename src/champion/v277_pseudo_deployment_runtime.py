"""Standalone prediction helper for the v271 pseudo-deployment XGB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def predict(
    frame: pd.DataFrame,
    asset: Path,
    base_runtime: object,
    base_asset: Path,
) -> np.ndarray:
    features = base_runtime.build(frame, base_asset)
    columns = json.loads((asset / "feature_columns.json").read_text(encoding="utf-8"))
    features = features.reindex(columns=columns)
    model = xgb.XGBClassifier()
    model.load_model(asset / "pseudo_deployment_xgb.json")
    return model.predict_proba(features)[:, 1].astype(np.float64)
