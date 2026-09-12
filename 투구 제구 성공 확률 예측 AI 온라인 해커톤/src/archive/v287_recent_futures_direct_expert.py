"""Audit a recent-era direct expert for the game_type=F process.

The F target regime changes sharply in 2023.  This experiment is deliberately
separate from the regular-season fallback family: a compact CatBoost direct
classifier is fitted only on recent F rows, then blended into the frozen v244
parent only on F rows.  The two contracts are early-2023 -> late-2023 and
full-2023 -> full-2024.  No test rows are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V287_RECENT_FUTURES_DIRECT_EXPERT_V1"
TARGET = "control_success"
CAT_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]
DROP_COLUMNS = ["row_id", TARGET, "season"]
WEIGHTS = (0.10, 0.25, 0.50, 0.75, 1.00)
PARAMS = {
    "iterations": 700,
    "depth": 7,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "l2_leaf_reg": 30.0,
    "random_strength": 0.35,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "one_hot_max_size": 16,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 16,
}


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    for column in CAT_COLUMNS:
        output[column] = output[column].fillna("__NA__").astype(str)
    for column in output.columns.difference(CAT_COLUMNS):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(-999.0)

    output["count_code"] = (
        frame["balls_before"].astype(str)
        + "-"
        + frame["strikes_before"].astype(str)
    )
    output["same_hand"] = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).astype(str)
    output["pressure_code"] = (
        frame["num_runners_on"].gt(0) | frame["li"].ge(1.5)
    ).astype(str)
    for column in ("count_code", "same_hand", "pressure_code"):
        output[column] = output[column].fillna("__NA__").astype(str)
    return output


def _fit_predict(
    fit_rows: pd.DataFrame,
    audit_rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, CatBoostClassifier]:
    fit_features = build_features(fit_rows)
    audit_features = build_features(audit_rows).reindex(columns=fit_features.columns)
    cat_columns = [
        column
        for column in CAT_COLUMNS + ["count_code", "same_hand", "pressure_code"]
        if column in fit_features.columns
    ]
    model = CatBoostClassifier(**PARAMS, random_seed=seed)
    model.fit(
        fit_features,
        fit_rows[TARGET].to_numpy(np.int8),
        cat_features=cat_columns,
    )
    return model.predict_proba(audit_features)[:, 1].astype(np.float64), model


def run(
    train_csv: Path,
    v285_axes: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    futures = train["game_type"].astype(str).eq("F")

    early_2023 = train.loc[
        train["season"].eq(2023) & futures & train["game_month"].lt(8)
    ].reset_index(drop=True)
    late_2023 = train.loc[
        train["season"].eq(2023) & futures & train["game_month"].ge(8)
    ].reset_index(drop=True)
    all_2023 = train.loc[train["season"].eq(2023) & futures].reset_index(drop=True)
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    futures_2024 = full_2024.loc[
        full_2024["game_type"].astype(str).eq("F")
    ].reset_index(drop=True)

    with np.load(v285_axes, allow_pickle=False) as saved:
        baseline_late_all = saved["baseline_late_2023"].astype(np.float64)
        baseline_2024 = saved["baseline_full_2024"].astype(np.float64)
    late_all = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    late_mask = late_all["game_type"].astype(str).eq("F").to_numpy()
    mask_2024 = full_2024["game_type"].astype(str).eq("F").to_numpy()
    if len(baseline_late_all) != len(late_all) or len(baseline_2024) != len(full_2024):
        raise ValueError("v285 axis alignment mismatch")

    prediction_late, source_model = _fit_predict(early_2023, late_2023, seed)
    prediction_2024, transfer_model = _fit_predict(all_2023, futures_2024, seed + 1)
    source_model.save_model(output_dir / "early23_to_late23_futures.cbm")
    transfer_model.save_model(output_dir / "full23_to_full24_futures.cbm")
    np.save(output_dir / "expert_late_2023.npy", prediction_late.astype(np.float32))
    np.save(output_dir / "expert_full_2024.npy", prediction_2024.astype(np.float32))

    axes = {
        "late_2023": (late_all, baseline_late_all, late_mask, prediction_late),
        "full_2024": (full_2024, baseline_2024, mask_2024, prediction_2024),
    }
    details: dict[str, dict[str, Any]] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for weight in WEIGHTS:
        key = f"w{weight:g}"
        per_axis: dict[str, Any] = {}
        for axis, (frame, baseline, active, expert) in axes.items():
            candidate = baseline.copy()
            candidate[active] = np.clip(
                baseline[active] + weight * (expert - baseline[active]),
                0.001,
                0.999,
            )
            candidates[(key, axis)] = candidate
            metrics = paired_metrics(
                {
                    "target": frame[TARGET].to_numpy(np.float64),
                    "exact_mask": np.ones(len(frame), dtype=bool),
                    "game_month": frame["game_month"].to_numpy(),
                },
                baseline,
                candidate,
                active,
            )
            per_axis[axis] = metrics
        rows.append(
            {
                "key": key,
                "weight": weight,
                "source_gain": per_axis["late_2023"]["gain"],
                "locked_gain": per_axis["full_2024"]["gain"],
            }
        )
        details[key] = per_axis

    # Choose only on the predeclared source contract; 2024 remains confirmatory.
    selected = max(rows, key=lambda row: row["source_gain"])
    selected_key = str(selected["key"])
    selected_candidate = candidates[(selected_key, "full_2024")]
    robustness = _robustness(
        {
            "target": full_2024[TARGET].to_numpy(np.float64),
            "pitcher_id": full_2024["pitcher_id"].to_numpy(),
            "batter_id": full_2024["batter_id"].to_numpy(),
        },
        baseline_2024,
        selected_candidate,
        mask_2024,
        [candidates[(f"w{weight:g}", "full_2024")] for weight in WEIGHTS],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    numeric_promote = bool(
        selected["source_gain"] > 0.0
        and selected["locked_gain"] >= 4.0
        and robust_pass
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "numeric_promote" if numeric_promote else "rejected",
        "rows": {
            "early_2023_f": len(early_2023),
            "late_2023_f": len(late_2023),
            "full_2023_f": len(all_2023),
            "full_2024_f": len(futures_2024),
        },
        "model_params": PARAMS,
        "selected": selected,
        "selected_metrics": details[selected_key],
        "all_weights": rows,
        "locked_robustness": robustness,
        "numeric_promotion_gate_passed": numeric_promote,
        "restrictions": {
            "official_train_only": True,
            "recent_futures_process_only": True,
            "source_selection_early23_to_late23_only": True,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2870)
    args = parser.parse_args()
    result = run(args.train_csv, args.v285_axes, args.output_dir, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
