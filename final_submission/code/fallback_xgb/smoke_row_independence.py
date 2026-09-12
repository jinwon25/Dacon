"""Row-independence smoke test for the fallback XGB asset.

The competition requires every evaluation row to be predicted from its own
inputs plus frozen training artifacts.  This drives the deployed transform three
ways over the same rows and requires the outputs to agree exactly:

  whole     all rows in one call
  singleton each row alone, so no neighbour can influence it
  shuffled  a permuted frame, restored to the original order afterwards

Any batch-level aggregate inside the transform would break at least one of these.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROTOCOL = "FALLBACK_XGB_ROW_INDEPENDENCE_V1"
SHUFFLE_SEED = 20260828
TOLERANCE = 1e-12


def _runtime_module():
    spec = importlib.util.spec_from_file_location(
        "fallback_xgb_frozen_runtime", ROOT / "fallback_xgb_frozen_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=ROOT.parent / "data" / "test.csv")
    parser.add_argument("--asset-dir", type=Path, default=ROOT)
    args = parser.parse_args()

    runtime = _runtime_module()
    rows = pd.read_csv(args.test_csv, encoding="utf-8-sig")

    whole = runtime.predict(rows, args.asset_dir)
    singleton = np.array(
        [runtime.predict(rows.iloc[[i]], args.asset_dir)[0] for i in range(len(rows))]
    )
    permuted = rows.sample(frac=1, random_state=SHUFFLE_SEED)
    restored = (
        pd.Series(runtime.predict(permuted, args.asset_dir), index=permuted.index)
        .reindex(rows.index)
        .to_numpy()
    )

    print({
        "rows": len(rows),
        "min": float(whole.min()),
        "max": float(whole.max()),
        "single_max_abs_diff": float(np.max(np.abs(whole - singleton))),
        "shuffle_max_abs_diff": float(np.max(np.abs(whole - restored))),
    })
    if not (
        np.allclose(whole, singleton, atol=TOLERANCE, rtol=0)
        and np.allclose(whole, restored, atol=TOLERANCE, rtol=0)
    ):
        raise SystemExit("row independence failed")
    print("row independence passed")


if __name__ == "__main__":
    main()
