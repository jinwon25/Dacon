"""Measure peak RSS for a scale proxy that exercises both v296 experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_v296_abs_regime_package import _run_with_peak_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=245789)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    positions = np.arange(args.rows) % len(source)
    proxy = source.iloc[positions].reset_index(drop=True).copy()
    proxy["row_id"] = [f"v296_mixed_scale_{index:06d}" for index in range(args.rows)]
    proxy.loc[proxy.index[1::2], "game_type"] = "F"
    result = _run_with_peak_memory(args.package, proxy, args.timeout_seconds)
    result["protocol"] = "AUDIT_V296_MIXED_RUNTIME_V1"
    result["forced_f_fraction"] = float(proxy["game_type"].eq("F").mean())
    result["forced_r_fraction"] = float(proxy["game_type"].eq("R").mean())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
