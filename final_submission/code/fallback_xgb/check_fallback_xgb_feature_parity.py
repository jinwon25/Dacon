"""Check that the frozen asset and the deployed transform agree on one contract.

Three things have to line up for the fallback component to be reproducible:

  1. the lookup tables rebuilt from `train.csv` match the shipped asset,
  2. the transform emits exactly the columns in `feature_columns.json`, in order,
  3. the booster was fit on that same column count.

Step 1 is the expensive one and needs the official `train.csv`; pass
`--skip-lookups` to check only the contract when the data is not mounted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
PROTOCOL = "FALLBACK_XGB_CONTRACT_PARITY_V1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT.parent / "data" / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT.parent / "data" / "test.csv")
    parser.add_argument("--asset-dir", type=Path, default=ROOT)
    parser.add_argument("--skip-lookups", action="store_true")
    args = parser.parse_args()

    runtime = _load("fallback_xgb_frozen_runtime", ROOT / "fallback_xgb_frozen_runtime.py")
    columns = json.loads((args.asset_dir / "feature_columns.json").read_text(encoding="utf-8"))
    frozen = joblib.load(args.asset_dir / "fallback_lookups.joblib")
    failures: list[str] = []

    if not args.skip_lookups:
        builder = _load("build_fallback_xgb_lookups", ROOT / "build_fallback_xgb_lookups.py")
        train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
        rebuilt = builder.build(train, frozen)
        if builder.verify(rebuilt, frozen):
            print("1. lookup 재생성 : 통과")
        else:
            failures.append("rebuilt lookups differ from the frozen asset")
            print("1. lookup 재생성 : 실패")
    else:
        print("1. lookup 재생성 : 건너뜀")

    rows = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    features = runtime.build(rows, args.asset_dir)
    if list(features.columns) == columns:
        print(f"2. 피처 계약     : 통과 ({len(columns)}열, 순서 일치)")
    else:
        failures.append("transform columns do not match feature_columns.json")
        print("2. 피처 계약     : 실패")
    if features.to_numpy().dtype != np.float32:
        failures.append(f"transform emitted {features.to_numpy().dtype}, expected float32")

    booster = xgb.XGBClassifier()
    booster.load_model(args.asset_dir / "fallback_xgb.json")
    fitted = booster.get_booster().num_features()
    if fitted == len(columns):
        print(f"3. 모델 입력 폭  : 통과 ({fitted}열)")
    else:
        failures.append(f"booster expects {fitted} features, contract has {len(columns)}")
        print("3. 모델 입력 폭  : 실패")

    if failures:
        for message in failures:
            print("  -", message)
        raise SystemExit("feature parity failed")
    print("\nfeature parity passed")


if __name__ == "__main__":
    main()
