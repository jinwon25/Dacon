"""Audit v345 formula parity, protected routes, and row independence."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from src.audit_v343_transition_workload import pressure_balanced_frame


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def beta_balanced_frame(train_csv: Path, rows_per_route: int) -> pd.DataFrame:
    frame = pressure_balanced_frame(train_csv, rows_per_route)
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    core = np.flatnonzero(regular & ~anchor)
    if len(core) < 2:
        raise ValueError("runtime fixture requires two R_CORE rows")
    selected = core[::2]
    frame.loc[selected, "asof_pitcher_n"] = 500.0
    frame.loc[selected, "asof_pitcher_pitchmix_n"] = 500.0
    frame.loc[selected, "asof_pitcher_fastball_rate"] = 0.34
    frame.loc[selected, "asof_pitcher_breaking_rate"] = 0.33
    frame.loc[selected, "asof_pitcher_offspeed_rate"] = 0.33
    return frame


def predict_partitioned(module, frame: pd.DataFrame) -> np.ndarray:
    split = max(1, len(frame) // 2)
    return np.concatenate(
        [
            module.predict_dataframe(frame.iloc[:split].reset_index(drop=True)),
            module.predict_dataframe(frame.iloc[split:].reset_index(drop=True)),
        ]
    )


def run(
    v335_zip: Path,
    v343_zip: Path,
    v345_zip: Path,
    train_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v345_audit_") as temporary:
        root = Path(temporary)
        paths = {name: root / name for name in ("v335", "v343", "v345")}
        for path, archive_path in (
            (paths["v335"], v335_zip),
            (paths["v343"], v343_zip),
            (paths["v345"], v345_zip),
        ):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(path)
        v335 = load_module("audit_v335_runtime_for_v345", paths["v335"] / "script.py")
        v343 = load_module("audit_v343_runtime_for_v345", paths["v343"] / "script.py")
        v345 = load_module("audit_v345_runtime", paths["v345"] / "script.py")
        frame = beta_balanced_frame(train_csv, rows_per_route)
        baseline = v335.predict_dataframe(frame.copy())
        parent = v343.predict_dataframe(frame.copy())
        candidate = v345.predict_dataframe(frame.copy())
        active = v345._beta_cell_mask(frame)
        beta = v345._predict_beta_cell(frame.loc[active].reset_index(drop=True))
        expected = parent.copy()
        expected[active] = np.clip(
            parent[active] + 0.10 * (beta - baseline[active]),
            0.001,
            0.999,
        )
        order = frame.sample(frac=1.0, random_state=345).index.to_numpy()
        shuffled = v345.predict_dataframe(frame.iloc[order].reset_index(drop=True))
        restored = np.empty_like(shuffled)
        restored[order] = shuffled
        partitioned = predict_partitioned(v345, frame)
        changed = np.abs(candidate - parent)
        protected = ~active
        result = {
            "protocol": "V345_TRANSITION_WORKLOAD_BETA_RUNTIME_AUDIT_V1",
            "rows": int(len(frame)),
            "beta_active_rows": int(active.sum()),
            "beta_changed_rows": int(np.count_nonzero(changed[active] > 0.0)),
            "formula_max_abs": float(np.max(np.abs(candidate - expected))),
            "protected_route_parity_max_abs": float(
                np.max(np.abs(candidate[protected] - parent[protected]))
            ),
            "shuffle_max_abs": float(np.max(np.abs(candidate - restored))),
            "partition_max_abs": float(np.max(np.abs(candidate - partitioned))),
            "finite": bool(np.isfinite(candidate).all()),
            "in_range": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_csv_read": train_csv.name.lower() == "test.csv",
            "test_aggregate_used": False,
            "test_used_for_candidate_selection": False,
        }
        result["status"] = "pass" if (
            result["beta_active_rows"] > 0
            and result["beta_changed_rows"] == result["beta_active_rows"]
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
    parser.add_argument("--v335-zip", type=Path, required=True)
    parser.add_argument("--v343-zip", type=Path, required=True)
    parser.add_argument("--v345-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.v335_zip,
                args.v343_zip,
                args.v345_zip,
                args.train_csv,
                args.output_json,
                args.rows_per_route,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
