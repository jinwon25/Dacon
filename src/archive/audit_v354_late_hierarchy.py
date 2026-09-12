"""Audit v354 formula parity, protected routes, and row independence."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from src.audit_v343_transition_workload import pressure_balanced_frame


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def maturity_balanced_frame(input_csv: Path, rows_per_route: int):
    frame = pressure_balanced_frame(input_csv, rows_per_route)
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    indices = np.flatnonzero(anchor)
    if len(indices) < 2:
        raise ValueError("v354 fixture requires two R_ANCHOR rows")
    frame.loc[indices[::2], "game_month"] = 8
    frame.loc[indices[1::2], "game_month"] = 7
    return frame


def predict_partitioned(module, frame) -> np.ndarray:
    split = max(1, len(frame) // 2)
    return np.concatenate(
        [
            module.predict_dataframe(frame.iloc[:split].reset_index(drop=True)),
            module.predict_dataframe(frame.iloc[split:].reset_index(drop=True)),
        ]
    )


def run(
    v353_zip: Path,
    v354_zip: Path,
    input_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v354_audit_") as temporary:
        root = Path(temporary)
        old = root / "v353"
        new = root / "v354"
        with zipfile.ZipFile(v353_zip) as archive:
            archive.extractall(old)
        with zipfile.ZipFile(v354_zip) as archive:
            archive.extractall(new)
        v353 = load_module("audit_v353_runtime_for_v354", old / "script.py")
        v354 = load_module("audit_v354_runtime", new / "script.py")
        frame = maturity_balanced_frame(input_csv, rows_per_route)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        anchor = regular & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
        month = frame["game_month"].to_numpy(np.int16)
        active = anchor & (month >= 8)

        parent = v353.predict_dataframe(frame.copy())
        candidate = v354.predict_dataframe(frame.copy())
        hierarchy = v354._predict_late_hierarchy(
            frame.loc[active].reset_index(drop=True)
        )
        expected = parent.copy()
        expected[active] = np.clip(
            0.80 * parent[active] + 0.20 * hierarchy, 0.001, 0.999
        )
        order = frame.sample(frac=1.0, random_state=354).index.to_numpy()
        shuffled = v354.predict_dataframe(frame.iloc[order].reset_index(drop=True))
        restored = np.empty_like(shuffled)
        restored[order] = shuffled
        partitioned = predict_partitioned(v354, frame)
        changed = np.abs(candidate - parent)
        protected = ~active
        result = {
            "protocol": "V354_LATE_HIERARCHY_RUNTIME_AUDIT_V1",
            "rows": int(len(frame)),
            "hierarchy_active_rows": int(active.sum()),
            "hierarchy_changed_rows": int(np.count_nonzero(changed[active] > 0.0)),
            "formula_max_abs": float(np.max(np.abs(candidate - expected))),
            "protected_route_parity_max_abs": float(
                np.max(np.abs(candidate[protected] - parent[protected]))
            ),
            "shuffle_max_abs": float(np.max(np.abs(candidate - restored))),
            "partition_max_abs": float(np.max(np.abs(candidate - partitioned))),
            "finite": bool(np.isfinite(candidate).all()),
            "in_range": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
            "runtime_seconds": float(time.perf_counter() - started),
            "input_csv": input_csv.name,
            "test_aggregate_used": False,
            "test_used_for_candidate_selection": False,
        }
        result["status"] = "pass" if (
            result["hierarchy_active_rows"] > 0
            and result["hierarchy_changed_rows"] == result["hierarchy_active_rows"]
            and result["formula_max_abs"] <= 1e-15
            and result["protected_route_parity_max_abs"] <= 1e-15
            and result["shuffle_max_abs"] <= 1e-15
            and result["partition_max_abs"] <= 1e-15
            and result["finite"]
            and result["in_range"]
        ) else "fail"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v353-zip", type=Path, required=True)
    parser.add_argument("--v354-zip", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.v353_zip, args.v354_zip, args.input_csv, args.output_json, args.rows_per_route), indent=2))


if __name__ == "__main__":
    main()
