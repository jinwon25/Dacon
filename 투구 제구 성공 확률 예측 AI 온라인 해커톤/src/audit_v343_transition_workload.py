"""Audit v343 runtime parity, protected routes, and row independence."""

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

from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.audit_v334_aggressive_moe import balanced_frame


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pressure_balanced_frame(train_csv: Path, rows_per_route: int) -> pd.DataFrame:
    frame = balanced_frame(train_csv, rows_per_route)
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    core_index = np.flatnonzero(regular & ~anchor)
    if len(core_index) < 2:
        raise ValueError("runtime fixture requires at least two R_CORE rows")
    pressure_index = core_index[::2]
    quiet_index = core_index[1::2]
    frame.loc[pressure_index, "num_runners_on"] = 1
    frame.loc[pressure_index, "li"] = 2.0
    frame.loc[quiet_index, "num_runners_on"] = 0
    frame.loc[quiet_index, "li"] = 1.0
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
    v334_zip: Path,
    v343_zip: Path,
    train_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v343_audit_") as temporary:
        root = Path(temporary)
        old = root / "v334"
        new = root / "v343"
        with zipfile.ZipFile(v334_zip) as archive:
            archive.extractall(old)
        with zipfile.ZipFile(v343_zip) as archive:
            archive.extractall(new)
        v334 = load_module("audit_v334_runtime_for_v343", old / "script.py")
        v343 = load_module("audit_v343_runtime", new / "script.py")
        frame = pressure_balanced_frame(train_csv, rows_per_route)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        anchor = regular & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
        rcore = regular & ~anchor
        futures = frame["game_type"].astype(str).eq("F").to_numpy()

        base = v334.predict_dataframe(frame.copy())
        candidate = v343.predict_dataframe(frame.copy())
        parent, h1, c3, active, _v334_duplicate = v334.predict_components(
            frame.copy()
        )
        base_active = v334._active_mask(frame)
        c3_base = v334._predict_c3(frame, v334.MEAN_RECENT_BASE_WEIGHT)
        bridge_parent = v334._predict_bridge_parent(frame)
        effective_bridge = parent + v334.BRIDGE_SCALE * (bridge_parent - parent)
        jy_probability = parent.copy()
        jy_probability[base_active] = np.clip(
            (1.0 - v334.H1_BASE_WEIGHT) * parent[base_active]
            + v334.H1_BASE_WEIGHT * h1[base_active]
            + v334.C3_WEIGHT * c3_base[base_active],
            0.001,
            0.999,
        )
        jy_probability[active] = np.clip(
            (1.0 - v334.H1_WEIGHT) * effective_bridge[active]
            + v334.H1_WEIGHT * h1[active]
            + v334.C3_WEIGHT * c3[active],
            0.001,
            0.999,
        )
        workload_active = active & rcore
        workload_h1 = v343._predict_workload_h1(
            frame.loc[workload_active].reset_index(drop=True)
        )
        workload_proposal = np.clip(
            0.82 * effective_bridge[workload_active]
            + 0.18 * workload_h1
            + v334.C3_WEIGHT * c3[workload_active],
            0.001,
            0.999,
        )
        expected = base.copy()
        expected[workload_active] = np.clip(
            base[workload_active]
            + workload_proposal
            - jy_probability[workload_active],
            0.001,
            0.999,
        )

        training_workload = workload_feature_frame(frame)
        runtime_workload = v343._attach_workload_features(frame)
        workload_feature_max_abs = float(
            np.nanmax(
                np.abs(
                    runtime_workload[training_workload.columns].to_numpy(np.float64)
                    - training_workload.to_numpy(np.float64)
                )
            )
        )
        order = frame.sample(frac=1.0, random_state=343).index.to_numpy()
        shuffled_prediction = v343.predict_dataframe(
            frame.iloc[order].reset_index(drop=True)
        )
        restored = np.empty_like(shuffled_prediction)
        restored[order] = shuffled_prediction
        partitioned = predict_partitioned(v343, frame)

        inactive_core = rcore & ~workload_active
        protected = futures | anchor | inactive_core
        changed = np.abs(candidate[workload_active] - base[workload_active])
        input_is_test = train_csv.name.lower() == "test.csv"
        result = {
            "protocol": "V343_TRANSITION_WORKLOAD_RUNTIME_AUDIT_V1",
            "rows": int(len(frame)),
            "route_rows": {
                "F": int(futures.sum()),
                "R_ANCHOR": int(anchor.sum()),
                "R_CORE_PRESSURE": int(workload_active.sum()),
                "R_CORE_NONPRESSURE": int(inactive_core.sum()),
            },
            "formula_max_abs": float(np.max(np.abs(candidate - expected))),
            "protected_route_parity_max_abs": float(
                np.max(np.abs(candidate[protected] - base[protected]))
            ),
            "workload_feature_formula_max_abs": workload_feature_max_abs,
            "workload_changed_rows": int(np.count_nonzero(changed > 0.0)),
            "workload_mean_abs_shift": float(np.mean(changed)),
            "shuffle_max_abs": float(np.max(np.abs(candidate - restored))),
            "partition_max_abs": float(np.max(np.abs(candidate - partitioned))),
            "finite": bool(np.isfinite(candidate).all()),
            "in_range": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_csv_read": input_is_test,
            "test_use_scope": (
                "unlabelled row-local runtime fixtures only; no aggregate, "
                "target, tuning or candidate selection"
                if input_is_test
                else "not used"
            ),
            "test_aggregate_used": False,
            "test_used_for_candidate_selection": False,
        }
        result["status"] = "pass" if (
            result["formula_max_abs"] <= 1e-15
            and result["protected_route_parity_max_abs"] <= 1e-15
            and result["workload_feature_formula_max_abs"] <= 1e-12
            and result["workload_changed_rows"] == result["route_rows"]["R_CORE_PRESSURE"]
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
    parser.add_argument("--v334-zip", type=Path, required=True)
    parser.add_argument("--v343-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(
        args.v334_zip,
        args.v343_zip,
        args.train_csv,
        args.output_json,
        args.rows_per_route,
    ), indent=2))


if __name__ == "__main__":
    main()
