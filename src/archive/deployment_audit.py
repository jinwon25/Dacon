"""Offline package, row-order independence, and runtime audit for v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("audited_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import package script")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def run(project: Path) -> dict:
    zip_path = project / "submit_v2.zip"
    test = pd.read_csv(project / "data/test.csv", encoding="utf-8-sig")
    benchmark = pd.read_csv(project / "data/train.csv", encoding="utf-8-sig", nrows=245_789, low_memory=False).drop(columns="control_success")
    with tempfile.TemporaryDirectory(prefix="v2_deploy_audit_") as temp_name:
        root = Path(temp_name)
        with zipfile.ZipFile(zip_path) as archive: archive.extractall(root)
        module = _load_module(root / "script.py")
        baseline = np.asarray(module.predict_dataframe(test), dtype=float)
        order_checks = {}
        reverse = np.asarray(module.predict_dataframe(test.iloc[::-1].reset_index(drop=True)), dtype=float)[::-1]
        order_checks["reverse_order_max_abs"] = float(np.max(np.abs(baseline - reverse)))
        shuffled_order = np.random.default_rng(42).permutation(len(test))
        shuffled = np.asarray(module.predict_dataframe(test.iloc[shuffled_order].reset_index(drop=True)), dtype=float)
        restored = np.empty_like(shuffled); restored[shuffled_order] = shuffled
        order_checks["shuffled_order_max_abs"] = float(np.max(np.abs(baseline - restored)))
        chunks = np.concatenate([module.predict_dataframe(test.iloc[start:start + 2].reset_index(drop=True)) for start in range(0, len(test), 2)])
        order_checks["chunked_max_abs"] = float(np.max(np.abs(baseline - chunks)))
        start = time.perf_counter(); before = psutil.Process().memory_info().rss
        prediction = np.asarray(module.predict_dataframe(benchmark), dtype=float)
        seconds = time.perf_counter() - start; rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        static = (root / "script.py").read_text(encoding="utf-8").lower()
        offline = not any(token in static for token in ("requests.", "urllib.request", "http://", "https://", "socket."))
        row_local = not any(token in static for token in ("groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding("))
    result = {
        "parent": "submit_v2.zip",
        "n_sample_rows": len(test),
        "n_benchmark_rows": len(benchmark),
        "finite_and_probability_range": bool(np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()),
        "order_checks": order_checks,
        "offline_static_check": offline,
        "row_local_static_check": row_local,
        "benchmark_seconds": seconds,
        "peak_rss_mb_observed": rss_mb,
        "parent_prediction_parity_note": "parent itself is the reference; correction-off candidate parity is not applicable because no candidate was promoted",
    }
    (project / "reports/deployment_gate_20260809.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (project / "reports/deployment_gate_20260809.md").write_text("# Deployment gate\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in result.items()) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__":
    main()
