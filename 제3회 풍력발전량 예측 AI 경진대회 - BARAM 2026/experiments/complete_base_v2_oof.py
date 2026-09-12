"""Complete the missing December 2024 Base-v2 OOF block.

The retained nested Base-v2 cache ends at 2024-12-01 00:00 because its
predeclared outer seasons stop at 2024-SON.  This module adds one causal
``2025-DJF`` fold: selection uses the preceding complete issue season, fitting
uses only issue cycles available before the new fold, and the resulting
December block is stitched to the retained cache without overlap.

Only the group-2 member needed by the public-incumbent overlay is rebuilt.
No submission is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.nested_quantile_base import (
    DEFAULT_SEEDS,
    Selection,
    _fit_fixed_lgb_ensemble,
    make_nested_folds,
)
from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, evaluate_group
from train import select_feature_columns


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_2"
OUTER_SEASON = "2025-DJF"


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stitch_prediction(
    full_index: pd.DatetimeIndex,
    retained_index: pd.DatetimeIndex,
    retained_prediction: np.ndarray,
    extension_index: pd.DatetimeIndex,
    extension_prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stitch two non-overlapping causal prediction blocks onto ``full_index``."""
    retained_prediction = np.asarray(retained_prediction, dtype=float)
    extension_prediction = np.asarray(extension_prediction, dtype=float)
    if len(retained_index) != len(retained_prediction):
        raise ValueError("retained index/prediction length mismatch")
    if len(extension_index) != len(extension_prediction):
        raise ValueError("extension index/prediction length mismatch")
    if retained_index.duplicated().any() or extension_index.duplicated().any():
        raise ValueError("OOF blocks contain duplicate timestamps")
    overlap = retained_index.intersection(extension_index)
    if len(overlap):
        raise ValueError("retained and extension OOF blocks overlap")

    output = pd.Series(np.nan, index=full_index, dtype=float)
    retained_rows = retained_index.intersection(full_index)
    extension_rows = extension_index.intersection(full_index)
    output.loc[retained_rows] = pd.Series(
        retained_prediction, index=retained_index
    ).loc[retained_rows]
    output.loc[extension_rows] = pd.Series(
        extension_prediction, index=extension_index
    ).loc[extension_rows]
    missing = output.index[output.isna()]
    if len(missing):
        raise ValueError(
            "completed OOF still has missing timestamps: "
            f"{missing.min()} .. {missing.max()} ({len(missing)} rows)"
        )
    return output.to_numpy(dtype=float), {
        "full_rows": int(len(full_index)),
        "retained_rows": int(len(retained_rows)),
        "extension_rows": int(len(extension_rows)),
        "overlap_rows": 0,
        "missing_rows": 0,
        "retained_start": retained_rows.min().isoformat(),
        "retained_end": retained_rows.max().isoformat(),
        "extension_start": extension_rows.min().isoformat(),
        "extension_end": extension_rows.max().isoformat(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _rooted(args.data_dir)
    cache_dir = _rooted(args.cache_dir)
    retained_path = _rooted(args.retained_predictions)
    retained_report_path = _rooted(args.retained_report)
    driver_path = _rooted(args.driver)
    output_cache = _rooted(args.output_cache)
    output_report = _rooted(args.output_report)

    features = load_or_build_features(data_dir, "train", cache_dir)
    labels = pd.read_csv(
        data_dir / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").reindex(features.index)
    issue_times = load_issue_times(
        data_dir / "train" / "gfs_train.csv", features.index
    )
    columns = select_feature_columns(features, TARGET, args.feature_set)
    x = features[columns]
    y = labels[TARGET].to_numpy(dtype=float)
    available = np.isfinite(y)
    folds = make_nested_folds(
        x.index,
        issue_times,
        available,
        purge_hours=args.purge_hours,
        outer_seasons=(OUTER_SEASON,),
        minimum_train_rows=2_000,
        minimum_valid_rows=500,
    )
    if len(folds) != 1:
        raise ValueError(f"expected exactly one {OUTER_SEASON} extension fold")
    fold = folds[0]

    # Reuse the production selection frozen from the four prior nested outer
    # seasons.  Re-running a fresh grid on the already inspected 2024 history
    # would add another selection degree of freedom and is unnecessary for
    # completing the missing causal block.
    retained_report = json.loads(
        retained_report_path.read_text(encoding="utf-8")
    )
    selected = Selection(
        **retained_report["groups"][TARGET]["final_selection"]
    )
    extension = _fit_fixed_lgb_ensemble(
        x,
        y,
        fold.train,
        fold.valid,
        CAPACITY_KWH[TARGET],
        selection=selected,
        seeds=DEFAULT_SEEDS,
        curtailment=None,
        n_jobs=args.n_jobs,
    )
    extension_index = x.index[fold.valid]
    extension_truth = y[fold.valid]

    retained = np.load(retained_path, allow_pickle=False)
    retained_index = pd.DatetimeIndex(
        pd.to_datetime(retained["index_ns"])
    )
    driver = np.load(driver_path, allow_pickle=False)
    full_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    full_truth = driver[f"{TARGET}__valid_truth"].astype(float)
    completed, coverage = stitch_prediction(
        full_index,
        retained_index,
        retained[f"{TARGET}__candidate"].astype(float),
        extension_index,
        extension,
    )
    aligned_extension_truth = pd.Series(
        extension_truth, index=extension_index
    ).reindex(full_index)
    truth_rows = aligned_extension_truth.notna().to_numpy()
    truth_error = np.abs(
        aligned_extension_truth.to_numpy(dtype=float)[truth_rows]
        - full_truth[truth_rows]
    )
    # The retained driver stores labels as float32.  A 1e-3 kWh tolerance is
    # smaller than one millionth of capacity and covers only that serialization
    # round-off; timestamps and values must otherwise agree.
    if float(truth_error.max(initial=0.0)) > 1e-3:
        raise ValueError("extension truth does not align with exact driver truth")

    train_issue = pd.DatetimeIndex(np.asarray(issue_times)[fold.train])
    valid_issue = pd.DatetimeIndex(np.asarray(issue_times)[fold.valid])
    issue_gap_hours = (
        valid_issue.min() - train_issue.max()
    ).total_seconds() / 3_600.0
    if issue_gap_hours < args.purge_hours:
        raise ValueError("extension fold violates the issue-time purge")

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_cache,
        index_ns=full_index.astype("int64").to_numpy(),
        issue_ns=pd.DatetimeIndex(
            pd.Series(issue_times, index=features.index).reindex(full_index)
        ).astype("int64").to_numpy(),
        **{
            f"{TARGET}__truth": full_truth.astype("float32"),
            f"{TARGET}__candidate": completed.astype("float32"),
            f"{TARGET}__extension_mask": np.asarray(
                full_index.isin(extension_index), dtype=bool
            ),
        },
    )
    extension_metric = evaluate_group(
        extension_truth, extension, CAPACITY_KWH[TARGET]
    )
    report = {
        "family": "causal_base_v2_oof_completion",
        "target": TARGET,
        "source": {
            "retained_predictions": retained_path.relative_to(ROOT).as_posix(),
            "retained_report": retained_report_path.relative_to(ROOT).as_posix(),
            "driver": driver_path.relative_to(ROOT).as_posix(),
        },
        "contract": {
            "outer_season": OUTER_SEASON,
            "selection": "frozen consensus from four retained outer seasons",
            "diagnostic_prior_season": fold.inner_name,
            "purge_hours": int(args.purge_hours),
            "outer_fold_used_for_selection": False,
            "public_score_used": False,
            "test_actual_generation_used": False,
        },
        "fold": {
            "train_rows": int(fold.train.sum()),
            "inner_train_rows": int(fold.inner_train.sum()),
            "inner_valid_rows": int(fold.inner_valid.sum()),
            "valid_rows": int(fold.valid.sum()),
            "train_issue_max": train_issue.max().isoformat(),
            "valid_issue_min": valid_issue.min().isoformat(),
            "issue_gap_hours": float(issue_gap_hours),
            "selected": asdict(selected),
            "retained_outer_selections": [
                item["selected_quantile"]
                for item in retained_report["groups"][TARGET]["folds"]
            ],
            "extension_metric": extension_metric.to_dict(),
        },
        "coverage": coverage,
        "output": {
            "path": output_cache.relative_to(ROOT).as_posix(),
            "sha256": _sha256(output_cache),
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--cache-dir", default="artifacts_final/feature_cache"
    )
    parser.add_argument(
        "--retained-predictions",
        default=(
            "artifacts_final/base_v2/group3_curtailment_full_20260725/"
            "predictions.npz"
        ),
    )
    parser.add_argument(
        "--retained-report",
        default=(
            "artifacts_final/base_v2/group3_curtailment_full_20260725/"
            "report.json"
        ),
    )
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument("--feature-set", default="own_idw")
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--output-cache",
        default="artifacts_final/lineage/base_v2_group2_complete_oof.npz",
    )
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "base_v2_group2_oof_completion_20260726.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "contract": report["contract"],
                "fold": report["fold"],
                "coverage": report["coverage"],
                "output": report["output"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
