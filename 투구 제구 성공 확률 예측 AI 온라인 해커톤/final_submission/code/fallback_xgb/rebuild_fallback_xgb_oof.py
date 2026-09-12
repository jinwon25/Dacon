"""RETIRED: the old full-train lookup path was not a valid OOF procedure.

Splitting estimator.fit rows was insufficient: the lookup tables were built
from the whole train, including the audit year's labels. main now fails closed
and points to the fold-local v242 diagnostic. No old OOF files are reused here.

The 2022 audit drops Futures rows from history because the league's earliest
seasons carry a different release-tracking regime.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
PROTOCOL = "FALLBACK_XGB_DEPLOYED_CONTRACT_OOF_V1"
YEARS = (2022, 2023, 2024)
HALF_LIFE_SEASONS = 2.0

PARAMS = dict(
    n_estimators=1800, learning_rate=.006, max_depth=10, min_child_weight=6000,
    subsample=.7, colsample_bytree=.5, reg_lambda=50., reg_alpha=1.,
    tree_method="hist", device="cuda:0", eval_metric="logloss", verbosity=0,
)
RANDOM_STATE = 2028


def _runtime_module():
    spec = importlib.util.spec_from_file_location(
        "fallback_xgb_frozen_runtime", ROOT / "fallback_xgb_frozen_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = target.mean()
    return 1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT.parent / "data" / "train.csv")
    parser.add_argument("--asset-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=PARAMS["device"])
    args = parser.parse_args()

    parser.error("Retired: all-train frozen lookups leak audit-season labels into OOF features. "
                 "Use python -m src.archive.v242_runtime_faithful_fallback_oof with "
                 "--train-csv, --trackman-csv, --pitcher-map-csv, --feature-columns-json, "
                 "--output-dir. Its fold-local outputs are a new diagnostic, not old-score replay.")


if __name__ == "__main__":
    main()
