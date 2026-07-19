"""Train LDAPS-only and GFS-only group-3 experts with honest forward OOF.

Best iteration is selected on 2023-Q4 after training on 2023-Q1-Q3.  The
retained OOF expert is then refit on all labelled 2023 and predicts 2024.
The supplied 2022 target column is empty and is never treated as training data. Q1/Q2/H2 metrics are
reported without using 2024 to tune the source models.  This module never
creates a submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.dynamic_router_oracle import load_experts, oracle_route
from src.metrics import CAPACITY_KWH, evaluate_group
from train import make_model


ROOT = Path(__file__).resolve().parents[1]
TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
OUTPUT_DIR = ROOT / "artifacts_final" / "nwp_source_ablation"
Q2_START = pd.Timestamp("2024-04-01")
H2_START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2025-01-01")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def source_columns(frame: pd.DataFrame, source: str) -> list[str]:
    calendar = [
        column
        for column in frame.columns
        if not column.startswith("ldaps__") and not column.startswith("gfs__")
    ]
    weather = [
        column
        for column in frame.columns
        if column.startswith(f"{source}__")
        and (
            "__kpx_group_" not in column
            or f"__{TARGET}__" in column
        )
    ]
    columns = calendar + weather
    if not weather:
        raise ValueError(f"No {source} features found")
    return columns


def train_source_expert(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    source: str,
    eligible_only: bool,
    valid_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = target.notna()
    tune_train = (
        (frame.index >= pd.Timestamp("2023-01-01"))
        & (frame.index < pd.Timestamp("2023-10-01"))
        & observed
    )
    tune_valid = (
        (frame.index >= pd.Timestamp("2023-10-01"))
        & (frame.index < pd.Timestamp("2024-01-01"))
        & observed
    )
    final_train = (
        (frame.index >= pd.Timestamp("2023-01-01"))
        & (frame.index < pd.Timestamp("2024-01-01"))
        & observed
    )
    if eligible_only:
        eligibility = target >= 0.10 * CAPACITY
        tune_train &= eligibility
        tune_valid &= eligibility
        final_train &= eligibility

    seed = 20_260 + (100 if source == "gfs" else 0) + int(eligible_only)
    tuning_model = make_model(seed, n_estimators=1600)
    tuning_model.fit(
        frame.loc[tune_train, columns],
        target.loc[tune_train],
        eval_set=[(frame.loc[tune_valid, columns], target.loc[tune_valid])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = max(100, int(tuning_model.best_iteration_))
    model = make_model(seed, n_estimators=best_iteration)
    model.fit(
        frame.loc[final_train, columns],
        target.loc[final_train],
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = np.clip(
        model.predict(frame.loc[valid_index, columns]),
        0.0,
        CAPACITY,
    )
    return prediction, {
        "source": source,
        "eligible_only": eligible_only,
        "seed": seed,
        "features": len(columns),
        "tune_train_rows": int(tune_train.sum()),
        "tune_valid_rows": int(tune_valid.sum()),
        "final_train_rows": int(final_train.sum()),
        "best_iteration": best_iteration,
    }


def _delta(before: Any, after: Any) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def main() -> None:
    features = pd.read_pickle(
        ROOT / "artifacts_final" / "feature_cache" / "features_train.pkl"
    )
    labels = pd.read_csv(
        ROOT / "data" / "train" / "train_labels.csv", encoding="utf-8-sig"
    )
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(features.index)
    index, truth, existing = load_experts()
    if not np.allclose(
        labels.loc[index, TARGET].to_numpy(dtype=float), truth, equal_nan=True
    ):
        raise ValueError("Cached exact truth differs from train labels")

    predictions: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        columns = source_columns(features, source)
        for eligible_only in (False, True):
            name = f"{source}_{'eligible' if eligible_only else 'all'}"
            predictions[name], specs[name] = train_source_expert(
                features,
                labels[TARGET],
                columns,
                source,
                eligible_only,
                index,
            )

    periods = {
        "q1": index < Q2_START,
        "q2": (index >= Q2_START) & (index < H2_START),
        "h2": (index >= H2_START) & (index < END),
    }
    base = existing["incumbent_finesweep"]
    metrics: dict[str, Any] = {}
    for name, prediction in predictions.items():
        metrics[name] = {}
        for period_name, rows in periods.items():
            before = evaluate_group(truth[rows], base[rows], CAPACITY)
            after = evaluate_group(truth[rows], prediction[rows], CAPACITY)
            metrics[name][period_name] = {
                "metric": after.to_dict(),
                "delta_vs_incumbent": _delta(before, after),
            }

    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    lead = ((index - issue) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    phase_id = np.asarray(
        [
            f"{issue_ns}:{int((lead_hour - 12) // 6)}"
            for issue_ns, lead_hour in zip(issue.astype("int64"), lead)
        ],
        dtype=object,
    )
    oracle_experts = {"incumbent_finesweep": base, **predictions}
    _, oracle = oracle_route(
        truth,
        oracle_experts,
        periods["h2"],
        phase_id,
    )

    eligible_h2 = periods["h2"] & (truth >= 0.10 * CAPACITY)
    error_matrix = np.column_stack(
        [truth[eligible_h2] - prediction[eligible_h2] for prediction in predictions.values()]
    )
    error_correlation = pd.DataFrame(
        np.corrcoef(error_matrix, rowvar=False),
        index=list(predictions),
        columns=list(predictions),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "validation_predictions.npz",
        index_ns=index.astype("int64").to_numpy(),
        truth=truth.astype("float32"),
        incumbent_finesweep=base.astype("float32"),
        **{name: value.astype("float32") for name, value in predictions.items()},
    )
    report = {
        "method": "forward LDAPS-only/GFS-only LightGBM source ablation",
        "contract": {
            "iteration_selection": "train 2023-Q1-Q3, early-stop on 2023-Q4",
            "oof_training": "refit labelled 2023, predict 2024",
            "2024_used_for_source_model_selection": False,
            "submission_created": False,
        },
        "specs": specs,
        "metrics": metrics,
        "h2_six_hour_oracle": oracle,
        "h2_eligible_error_correlation": error_correlation.to_dict(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "specs": specs,
                    "h2_deltas": {
                        name: value["h2"]["delta_vs_incumbent"]
                        for name, value in metrics.items()
                    },
                    "h2_six_hour_oracle": oracle,
                    "error_correlation": error_correlation.to_dict(),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
