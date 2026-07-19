import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.prediction_cache_diagnostics import analyze_cache
from src.metrics import CAPACITY_KWH


class PredictionCacheDiagnosticsTests(unittest.TestCase):
    def test_incomplete_month_is_excluded_from_stability(self) -> None:
        timestamps = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-03-01"]
        )
        cache = {
            "candidate_names": np.asarray(["baseline", "better"]),
            "test_index_ns": timestamps.astype("int64").to_numpy(),
        }
        for target, capacity in CAPACITY_KWH.items():
            truth = np.full(len(timestamps), 0.5 * capacity, dtype="float32")
            baseline = truth + 0.07 * capacity
            better = truth.copy()
            cache[f"{target}__valid_index_ns"] = timestamps.astype("int64").to_numpy()
            cache[f"{target}__valid_truth"] = truth
            cache[f"{target}__valid_matrix"] = np.column_stack([baseline, better]).astype("float32")
            cache[f"{target}__test_matrix"] = np.column_stack([baseline, better]).astype("float32")
            cache[f"{target}__selected_weights"] = np.asarray([0.0, 1.0], dtype="float32")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.npz"
            np.savez_compressed(cache_path, **cache)
            report = analyze_cache(cache_path, baseline_name="baseline", min_month_samples=2)

        for target_report in report["targets"].values():
            stability = target_report["stability_vs_baseline"]["selected_blend"]
            self.assertEqual(stability["months_total"], 2)
            self.assertEqual(stability["months_excluded"], 1)
            self.assertEqual(stability["months_improved"], 2)


if __name__ == "__main__":
    unittest.main()
