from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.blend_experiment import _search_weights
from src.features import TIME_COL
from src.metrics import CAPACITY_KWH, evaluate_competition


def _parse_cache(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Cache must use prefix=path format.")
    prefix, raw_path = value.split("=", 1)
    prefix = prefix.strip()
    if not prefix:
        raise argparse.ArgumentTypeError("Cache prefix cannot be empty.")
    return prefix, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", type=_parse_cache, required=True)
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--artifact-dir", default="artifacts_combined")
    parser.add_argument("--output", default="artifacts_combined/combined_member.csv")
    parser.add_argument("--n-iter", type=int, default=50_000)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(prefix, path, np.load(path, allow_pickle=False)) for prefix, path in args.cache]
    try:
        reference_test_index = loaded[0][2]["test_index_ns"]
        for prefix, path, cache in loaded[1:]:
            if not np.array_equal(cache["test_index_ns"], reference_test_index):
                raise ValueError(f"Test index mismatch in {prefix}={path}")

        names = []
        for prefix, _, cache in loaded:
            names.extend([f"{prefix}__{name}" for name in cache["candidate_names"].tolist()])

        sample = pd.read_csv(args.sample, encoding="utf-8-sig")
        sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
        expected_index = pd.to_datetime(reference_test_index, unit="ns")
        if not sample[TIME_COL].equals(pd.Series(expected_index, name=TIME_COL)):
            raise ValueError("Cache test index does not match sample submission.")
        submission = sample.set_index(TIME_COL)
        report: dict[str, object] = {
            "caches": [{"prefix": prefix, "path": str(path)} for prefix, path, _ in loaded],
            "candidate_names": names,
            "targets": {},
        }
        output_cache: dict[str, np.ndarray] = {
            "candidate_names": np.asarray(names),
            "test_index_ns": reference_test_index,
        }
        valid_truth: dict[str, np.ndarray] = {}
        valid_predictions: dict[str, np.ndarray] = {}

        for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
            ref_index = loaded[0][2][f"{target}__valid_index_ns"]
            truth = loaded[0][2][f"{target}__valid_truth"].astype(float)
            valid_parts = []
            test_parts = []
            for prefix, path, cache in loaded:
                if not np.array_equal(cache[f"{target}__valid_index_ns"], ref_index):
                    raise ValueError(f"Validation index mismatch for {target} in {prefix}={path}")
                if not np.allclose(cache[f"{target}__valid_truth"], truth, rtol=0, atol=1e-3):
                    raise ValueError(f"Validation truth mismatch for {target} in {prefix}={path}")
                valid_parts.append(cache[f"{target}__valid_matrix"].astype(float))
                test_parts.append(cache[f"{target}__test_matrix"].astype(float))
            valid_matrix = np.column_stack(valid_parts)
            test_matrix = np.column_stack(test_parts)
            weights, metric = _search_weights(
                valid_matrix,
                truth,
                capacity,
                seed=42_000 + target_i,
                n_iter=args.n_iter,
            )
            pred_valid = np.clip(valid_matrix @ weights, 0, capacity)
            pred_test = np.clip(test_matrix @ weights, 0, capacity)
            valid_truth[target] = truth
            valid_predictions[target] = pred_valid
            submission[target] = pred_test
            report["targets"][target] = {
                "metric": metric,
                "selected_weights": {
                    name: float(weight)
                    for name, weight in zip(names, weights)
                    if weight > 1e-8
                },
            }
            output_cache[f"{target}__valid_index_ns"] = ref_index
            output_cache[f"{target}__valid_truth"] = truth.astype("float32")
            output_cache[f"{target}__valid_matrix"] = valid_matrix.astype("float32")
            output_cache[f"{target}__test_matrix"] = test_matrix.astype("float32")
            output_cache[f"{target}__selected_weights"] = weights.astype("float32")
            print(target, metric, report["targets"][target]["selected_weights"], flush=True)

        report["competition_metric"] = evaluate_competition(valid_truth, valid_predictions)
        output = submission.reset_index()[sample.columns]
        output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        (artifact_dir / "combined_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(artifact_dir / "prediction_cache.npz", **output_cache)
        print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
        print(f"Saved combined member to {output_path.resolve()}", flush=True)
    finally:
        for _, _, cache in loaded:
            cache.close()


if __name__ == "__main__":
    main()
