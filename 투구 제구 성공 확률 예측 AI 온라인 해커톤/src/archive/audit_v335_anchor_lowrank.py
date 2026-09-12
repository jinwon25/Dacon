"""Audit v335 formula and row independence against the v320 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from src.audit_v334_aggressive_moe import (
    balanced_frame,
    load_module,
    predict_partitioned,
)


def run(
    v320_zip: Path,
    v335_zip: Path,
    input_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v335_audit_") as temporary:
        root = Path(temporary)
        old, new = root / "v320", root / "v335"
        with zipfile.ZipFile(v320_zip) as archive:
            archive.extractall(old)
        with zipfile.ZipFile(v335_zip) as archive:
            archive.extractall(new)
        v320 = load_module("audit_v320_runtime_for_v335", old / "script.py")
        v335 = load_module("audit_v335_runtime", new / "script.py")
        frame = balanced_frame(input_csv, rows_per_route)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        anchor = regular & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
        rcore = regular & ~anchor
        futures = frame["game_type"].astype(str).eq("F").to_numpy()
        parent = v320.predict_dataframe(frame.copy())
        candidate = v335.predict_dataframe(frame.copy())
        lowrank = v335._predict_futures_lowrank(frame.copy())
        expected = parent.copy()
        expected[anchor] = np.clip(
            expected[anchor] + 0.50 * lowrank[anchor], 0.001, 0.999
        )
        order = frame.sample(frac=1.0, random_state=335).index.to_numpy()
        shuffled = v335.predict_dataframe(frame.iloc[order].reset_index(drop=True))
        restored = np.empty_like(shuffled)
        restored[order] = shuffled
        partitioned = predict_partitioned(v335, frame)
        result = {
            "protocol": "V335_ANCHOR_LOWRANK_RUNTIME_AUDIT_V1",
            "rows": int(len(frame)),
            "route_rows": {
                "F": int(futures.sum()),
                "R_ANCHOR": int(anchor.sum()),
                "R_CORE": int(rcore.sum()),
            },
            "f_parity_max_abs": float(np.max(np.abs(candidate[futures] - parent[futures]))),
            "rcore_parity_max_abs": float(np.max(np.abs(candidate[rcore] - parent[rcore]))),
            "anchor_formula_max_abs": float(np.max(np.abs(candidate[anchor] - expected[anchor]))),
            "shuffle_max_abs": float(np.max(np.abs(candidate - restored))),
            "partition_max_abs": float(np.max(np.abs(candidate - partitioned))),
            "finite": bool(np.isfinite(candidate).all()),
            "in_range": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_csv_read": input_csv.name.lower() == "test.csv",
            "test_use_scope": (
                "unlabelled row-local runtime fixtures only; no aggregate, "
                "target, tuning or candidate selection"
            ),
            "test_aggregate_used": False,
            "test_used_for_candidate_selection": False,
        }
        result["status"] = "pass" if (
            result["f_parity_max_abs"] <= 1e-15
            and result["rcore_parity_max_abs"] <= 1e-15
            and result["anchor_formula_max_abs"] <= 1e-15
            and result["shuffle_max_abs"] <= 1e-15
            and result["partition_max_abs"] <= 1e-15
            and result["finite"] and result["in_range"]
        ) else "fail"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v320-zip", type=Path, required=True)
    parser.add_argument("--v335-zip", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(
        args.v320_zip, args.v335_zip, args.input_csv,
        args.output_json, args.rows_per_route,
    ), indent=2))


if __name__ == "__main__":
    main()
