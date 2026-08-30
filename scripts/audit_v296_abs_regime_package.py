"""Deep mechanical and magnitude audit for the v296 ABS-regime package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostClassifier

from src.core.packaging import _safe_extract, run_package


ID_COL = "row_id"
TARGET = "control_success"
RMS_TARGET_1190 = 0.005752
WEIGHT = 0.10


def _logit(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET]).to_numpy(np.float64)


def _mixed_probe(frame: pd.DataFrame) -> pd.DataFrame:
    mixed = pd.concat([frame, frame], ignore_index=True)
    mixed[ID_COL] = [f"v296_mixed_{index:04d}" for index in range(len(mixed))]
    mixed.loc[mixed.index[::2], "game_type"] = "R"
    mixed.loc[mixed.index[1::2], "game_type"] = "F"
    return mixed


def _all_r_proxy(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    positions = np.arange(rows) % len(frame)
    output = frame.iloc[positions].reset_index(drop=True).copy()
    output[ID_COL] = [f"v296_all_r_{index:06d}" for index in range(rows)]
    output["game_type"] = "R"
    return output


def _run_with_peak_memory(
    package: Path, frame: pd.DataFrame, timeout_seconds: float
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v296_all_r_runtime_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, stage)
        data = stage / "data"
        data.mkdir()
        frame.to_csv(data / "test.csv", index=False)
        pd.DataFrame({ID_COL: frame[ID_COL], TARGET: 0.0}).to_csv(
            data / "sample_submission.csv", index=False
        )
        started = time.perf_counter()
        process = psutil.Popen(
            [sys.executable, "script.py"],
            cwd=stage,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        peak_rss = 0
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed > timeout_seconds:
                process.kill()
                raise TimeoutError(f"v296 all-R proxy exceeded {timeout_seconds}s")
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_rss = max(peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        runtime = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(f"v296 all-R run failed\nstdout={stdout}\nstderr={stderr}")
        output = pd.read_csv(stage / "output" / "submission.csv")
        probability = _aligned(frame, output)
        return {
            "rows": len(frame),
            "forced_r_fraction": 1.0,
            "runtime_seconds": float(runtime),
            "peak_rss_mb": float(peak_rss / (1024.0**2)),
            "finite": bool(np.isfinite(probability).all()),
            "in_range": bool(np.all((probability >= 0.0) & (probability <= 1.0))),
            "row_order_preserved": bool(output[ID_COL].tolist() == frame[ID_COL].tolist()),
            "mean_prediction": float(probability.mean()),
            "min_prediction": float(probability.min()),
            "max_prediction": float(probability.max()),
            "stdout": stdout.strip(),
            "stderr_warning_lines": int(sum(bool(line.strip()) for line in stderr.splitlines())),
        }


def run(
    parent_package: Path,
    candidate_package: Path,
    smoke_test_csv: Path,
    train_csv: Path,
    v290_axes: Path,
    output_json: Path,
    scale_rows: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    smoke = pd.read_csv(smoke_test_csv, encoding="utf-8-sig", low_memory=False)
    mixed = _mixed_probe(smoke)
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET: 0.0})
    parent_out, parent_seconds, _ = run_package(
        parent_package, mixed, sample, timeout=int(timeout_seconds)
    )
    candidate_out, candidate_seconds, _ = run_package(
        candidate_package, mixed, sample, timeout=int(timeout_seconds)
    )
    parent = _aligned(mixed, parent_out)
    candidate = _aligned(mixed, candidate_out)
    regular = mixed["game_type"].astype(str).eq("R").to_numpy()
    futures = mixed["game_type"].astype(str).eq("F").to_numpy()

    with tempfile.TemporaryDirectory(prefix="v296_formula_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(candidate_package) as archive:
            _safe_extract(archive, stage)
        runtime = _load_module(
            "v296_abs_runtime_audit", stage / "model" / "abs_regular" / "runtime.py"
        )
        expert = runtime.predict(mixed, stage / "model" / "abs_regular")
        expected = parent.copy()
        expected[regular] = _expit(
            _logit(parent[regular])
            + WEIGHT * (_logit(expert[regular]) - _logit(parent[regular]))
        )
        formula_error = float(np.max(np.abs(candidate[regular] - expected[regular])))

        # Magnitude only: the final expert is fitted on these same 2024 labels,
        # so this is not an honest performance validation.
        full_2024 = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
        full_2024 = full_2024.loc[full_2024["season"].eq(2024)].reset_index(drop=True)
        r_mask = full_2024["game_type"].astype(str).eq("R").to_numpy()
        with np.load(v290_axes, allow_pickle=False) as saved:
            historical_parent = saved["candidate_full_2024"].astype(np.float64)
        if len(historical_parent) != len(full_2024):
            raise ValueError("v290 historical proxy alignment mismatch")
        asset = stage / "model" / "abs_regular"
        columns = json.loads((asset / "feature_columns.json").read_text(encoding="utf-8"))
        features = runtime.build_features(
            full_2024.loc[r_mask].reset_index(drop=True), columns
        )
        seed_predictions = []
        for seed in runtime.SEEDS:
            model = CatBoostClassifier()
            model.load_model(asset / f"abs_regular_expert_seed{seed}.cbm")
            seed_predictions.append(model.predict_proba(features)[:, 1])
        seed_stack = np.column_stack(seed_predictions)
        ensemble = seed_stack.mean(axis=1)
        diagnostic = historical_parent.copy()
        diagnostic[r_mask] = _expit(
            _logit(historical_parent[r_mask])
            + WEIGHT * (_logit(ensemble) - _logit(historical_parent[r_mask]))
        )
        shift = diagnostic - historical_parent
        rms = float(np.sqrt(np.mean(np.square(shift))))
        magnitude = {
            "scope": "2024 official-train feature proxy; expert predictions are in-sample",
            "honest_performance_validation": False,
            "rows": len(full_2024),
            "active_r_rows": int(r_mask.sum()),
            "active_fraction": float(r_mask.mean()),
            "whole_row_rms_shift": rms,
            "target_1190_rms": RMS_TARGET_1190,
            "target_rms_ratio": float(rms / RMS_TARGET_1190),
            "active_rms_shift": float(np.sqrt(np.mean(np.square(shift[r_mask])))),
            "active_mean_abs_shift": float(np.mean(np.abs(shift[r_mask]))),
            "active_max_abs_shift": float(np.max(np.abs(shift[r_mask]))),
            "ensemble_prediction_mean": float(ensemble.mean()),
            "ensemble_prediction_std": float(ensemble.std()),
            "seed_row_std_mean": float(seed_stack.std(axis=1).mean()),
            "seed_row_std_p95": float(np.quantile(seed_stack.std(axis=1), 0.95)),
        }

    all_r_runtime = _run_with_peak_memory(
        candidate_package, _all_r_proxy(smoke, scale_rows), timeout_seconds
    )
    result = {
        "protocol": "AUDIT_V296_ABS_REGIME_PACKAGE_V1",
        "mixed_probe": {
            "rows": len(mixed),
            "parent_runtime_seconds": float(parent_seconds),
            "candidate_runtime_seconds": float(candidate_seconds),
            "f_exact_parity_max_abs_diff": float(
                np.max(np.abs(candidate[futures] - parent[futures]))
            ),
            "r_formula_max_abs_diff": formula_error,
            "r_changed_rows": int(np.sum(np.abs(candidate[regular] - parent[regular]) > 0.0)),
            "r_rows": int(regular.sum()),
        },
        "magnitude_diagnostic": magnitude,
        "all_r_scale_runtime": all_r_runtime,
        "gates": {
            "f_exact_parity": bool(
                np.max(np.abs(candidate[futures] - parent[futures])) == 0.0
            ),
            "r_formula": bool(formula_error <= 1e-12),
            "r_nontrivial": bool(np.any(np.abs(candidate[regular] - parent[regular]) > 0.0)),
            "whole_row_rms_exceeds_optimal_dose_identity_reference": bool(
                rms >= RMS_TARGET_1190
            ),
            "runtime_under_600_seconds": bool(all_r_runtime["runtime_seconds"] < 600.0),
            "finite": bool(all_r_runtime["finite"]),
            "in_range": bool(all_r_runtime["in_range"]),
            "row_order": bool(all_r_runtime["row_order_preserved"]),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-package", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--smoke-test-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scale-rows", type=int, default=245789)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.parent_package,
                args.candidate_package,
                args.smoke_test_csv,
                args.train_csv,
                args.v290_axes,
                args.output_json,
                args.scale_rows,
                args.timeout_seconds,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
