"""CatBoost source-isolated experts under the same forward contract.

Only eligible-target training is used because the LightGBM source ablation
showed it dominates the all-row variants for both NWP sources.  Iterations are
selected on 2023-Q4 and the 2024 surface is never used for model tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.dynamic_router_oracle import load_experts, oracle_route
from experiments.nwp_source_ablation import (
    CAPACITY,
    END,
    H2_START,
    OUTPUT_DIR,
    Q2_START,
    ROOT,
    TARGET,
    _delta,
    _to_builtin,
    source_columns,
)
from experiments.blocked_rolling_validation import load_issue_times
from src.metrics import evaluate_group
from train import make_catboost_model


CAT_OUTPUT = ROOT / "artifacts_final" / "nwp_source_catboost"


def fit_source(
    features: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    source: str,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = target.notna()
    eligible = target >= 0.10 * CAPACITY
    tune_train = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < pd.Timestamp("2023-10-01"))
        & observed
        & eligible
    )
    tune_valid = (
        (features.index >= pd.Timestamp("2023-10-01"))
        & (features.index < pd.Timestamp("2024-01-01"))
        & observed
        & eligible
    )
    final_train = (
        (features.index >= pd.Timestamp("2023-01-01"))
        & (features.index < pd.Timestamp("2024-01-01"))
        & observed
        & eligible
    )
    seed = 30_260 + (100 if source == "gfs" else 0)
    tuning = make_catboost_model(seed, iterations=1800)
    tuning.fit(
        features.loc[tune_train, columns],
        target.loc[tune_train],
        eval_set=(features.loc[tune_valid, columns], target.loc[tune_valid]),
        use_best_model=True,
    )
    best_iteration = max(100, int(tuning.get_best_iteration() + 1))
    model = make_catboost_model(seed, iterations=best_iteration)
    model.fit(features.loc[final_train, columns], target.loc[final_train])
    prediction = np.clip(
        model.predict(features.loc[index, columns]), 0.0, CAPACITY
    )
    return prediction, {
        "source": source,
        "family": "catboost",
        "eligible_only": True,
        "seed": seed,
        "features": len(columns),
        "tune_train_rows": int(tune_train.sum()),
        "tune_valid_rows": int(tune_valid.sum()),
        "final_train_rows": int(final_train.sum()),
        "best_iteration": best_iteration,
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
    base = existing["incumbent_finesweep"]
    predictions: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        name = f"cat_{source}_eligible"
        predictions[name], specs[name] = fit_source(
            features,
            labels[TARGET],
            source_columns(features, source),
            source,
            index,
        )

    periods = {
        "q1": index < Q2_START,
        "q2": (index >= Q2_START) & (index < H2_START),
        "h2": (index >= H2_START) & (index < END),
    }
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
    _, oracle = oracle_route(
        truth,
        {"incumbent_finesweep": base, **predictions},
        periods["h2"],
        phase_id,
    )
    eligible_h2 = periods["h2"] & (truth >= 0.10 * CAPACITY)
    error_correlation = float(
        np.corrcoef(
            *[
                truth[eligible_h2] - prediction[eligible_h2]
                for prediction in predictions.values()
            ]
        )[0, 1]
    )

    CAT_OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CAT_OUTPUT / "validation_predictions.npz",
        index_ns=index.astype("int64").to_numpy(),
        truth=truth.astype("float32"),
        incumbent_finesweep=base.astype("float32"),
        **{name: value.astype("float32") for name, value in predictions.items()},
    )
    report = {
        "method": "forward CatBoost LDAPS/GFS isolated eligible experts",
        "contract": {
            "iteration_selection": "2023-Q1-Q3 train / 2023-Q4 validation",
            "oof_training": "all labelled 2023 predicts 2024",
            "submission_created": False,
        },
        "specs": specs,
        "metrics": metrics,
        "h2_six_hour_oracle": oracle,
        "cat_source_error_correlation": error_correlation,
    }
    (CAT_OUTPUT / "report.json").write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "specs": specs,
                    "h2_deltas": {
                        name: values["h2"]["delta_vs_incumbent"]
                        for name, values in metrics.items()
                    },
                    "h2_six_hour_oracle": oracle,
                    "cat_source_error_correlation": error_correlation,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {CAT_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
