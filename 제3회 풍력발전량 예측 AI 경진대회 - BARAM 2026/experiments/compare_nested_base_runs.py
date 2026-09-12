from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.nested_quantile_base import (
    _competition_delta,
    _issue_bootstrap,
    _ordered_issue_seasons,
)
from src.metrics import CAPACITY_KWH


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def compare_runs(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    issue_times: pd.DatetimeIndex,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    required = {"index_ns", "test_index_ns"}
    for group in CAPACITY_KWH:
        required.update(
            {
                f"{group}__truth",
                f"{group}__candidate",
                f"{group}__test",
            }
        )
    for name, cache in (("reference", reference), ("candidate", candidate)):
        missing = sorted(required - set(cache))
        if missing:
            raise ValueError(f"{name} cache is missing keys: {missing}")

    if not np.array_equal(reference["index_ns"], candidate["index_ns"]):
        raise ValueError("Reference and candidate OOF timestamps differ")
    if not np.array_equal(reference["test_index_ns"], candidate["test_index_ns"]):
        raise ValueError("Reference and candidate test timestamps differ")
    if len(issue_times) != len(reference["index_ns"]):
        raise ValueError("Issue times and OOF cache have different row counts")

    truth: dict[str, np.ndarray] = {}
    reference_prediction: dict[str, np.ndarray] = {}
    candidate_prediction: dict[str, np.ndarray] = {}
    test_movement: dict[str, Any] = {}
    for group in CAPACITY_KWH:
        truth_key = f"{group}__truth"
        prediction_key = f"{group}__candidate"
        test_key = f"{group}__test"
        if not np.allclose(
            reference[truth_key],
            candidate[truth_key],
            rtol=0.0,
            atol=1e-5,
            equal_nan=True,
        ):
            raise ValueError(f"Reference and candidate truth differ for {group}")
        truth[group] = reference[truth_key].astype(float)
        reference_prediction[group] = reference[prediction_key].astype(float)
        candidate_prediction[group] = candidate[prediction_key].astype(float)
        movement = candidate[test_key].astype(float) - reference[test_key].astype(float)
        test_movement[group] = {
            "mean_absolute_kwh": float(np.mean(np.abs(movement))),
            "p95_absolute_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "maximum_absolute_kwh": float(np.max(np.abs(movement))),
            "changed_rows": int(np.count_nonzero(np.abs(movement) > 1e-6)),
        }

    seasons, _ = _ordered_issue_seasons(
        pd.to_datetime(reference["index_ns"]),
        issue_times,
    )
    overall = _competition_delta(truth, reference_prediction, candidate_prediction)
    seasonal: dict[str, Any] = {}
    for season in dict.fromkeys(seasons):
        rows = seasons == season
        seasonal[season] = _competition_delta(
            {group: values[rows] for group, values in truth.items()},
            {group: values[rows] for group, values in reference_prediction.items()},
            {group: values[rows] for group, values in candidate_prediction.items()},
        )
    bootstrap = _issue_bootstrap(
        truth,
        reference_prediction,
        candidate_prediction,
        issue_times,
        seasons,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    gates = {
        "score_positive": overall["delta"]["score"] > 0.0,
        "one_minus_nmae_nonnegative": overall["delta"]["one_minus_nmae"] >= 0.0,
        "ficr_nonnegative": overall["delta"]["ficr"] >= 0.0,
        "worst_season_nonnegative": min(
            result["delta"]["score"] for result in seasonal.values()
        )
        >= 0.0,
        "bootstrap_q05_nonnegative": bootstrap["q05"] >= 0.0,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.90,
    }
    exploratory_gates = {
        "score_minimum": overall["delta"]["score"] >= 0.00015,
        "one_minus_nmae_tolerable": overall["delta"]["one_minus_nmae"]
        >= -0.00035,
        "ficr_tolerable": overall["delta"]["ficr"] >= -0.00035,
        "worst_season_tolerable": min(
            result["delta"]["score"] for result in seasonal.values()
        )
        >= -0.00035,
        "bootstrap_q05_tolerable": bootstrap["q05"] >= -0.00025,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"] >= 0.80,
    }
    qualified = bool(all(gates.values()))
    decision_tier = (
        "candidate"
        if qualified
        else "exploratory"
        if all(exploratory_gates.values())
        else "rejected"
    )
    return {
        "outer_macro": overall,
        "seasonal_macro": seasonal,
        "issue_block_bootstrap": bootstrap,
        "promotion_gates": gates,
        "exploratory_gates": exploratory_gates,
        "decision_tier": decision_tier,
        "qualified": qualified,
        "test_movement": test_movement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument(
        "--gfs-train",
        default="data/train/gfs_train.csv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_725)
    args = parser.parse_args()

    reference_path = Path(args.reference_run) / "predictions.npz"
    candidate_path = Path(args.candidate_run) / "predictions.npz"
    reference = _load_cache(reference_path)
    candidate = _load_cache(candidate_path)
    index = pd.to_datetime(reference["index_ns"])
    issue_times = load_issue_times(Path(args.gfs_train), index)
    report = compare_runs(
        reference,
        candidate,
        issue_times,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    report["reference"] = {
        "path": reference_path.as_posix(),
        "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    report["candidate"] = {
        "path": candidate_path.as_posix(),
        "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
