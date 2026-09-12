"""Audit a Brier-aligned CatBoost regressor for the recent-F route.

The v290 F expert optimizes binary Logloss although the competition minimizes
squared probability error.  This experiment fits the same recent F windows
and feature recipe with CatBoost RMSE, then tests both a full replacement and
a fixed classifier/regressor expert average at the unchanged 10% route dose.
Selection is early-2023 -> late-2023 only; full-2024 remains locked.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.archive.v287_recent_futures_direct_expert import CAT_COLUMNS, PARAMS, TARGET, build_features
from src.archive.v301_multidomain_futures_expert import bss, compose, paired_metrics


PROTOCOL = "V303_FUTURES_BRIER_REGRESSOR_V1"
SEED = 3030
FUTURES_WEIGHT = 0.10
EXTRA_CAT_COLUMNS = ["count_code", "same_hand", "pressure_code"]
REGRESSOR_PARAMS = {
    key: value for key, value in PARAMS.items()
    if key not in {"loss_function"}
}
REGRESSOR_PARAMS.update({"loss_function": "RMSE"})


def fit_predict(
    fit_rows: pd.DataFrame,
    audit_rows: pd.DataFrame,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_features = build_features(fit_rows)
    audit_features = build_features(audit_rows).reindex(columns=fit_features.columns)
    categorical = [
        column for column in list(CAT_COLUMNS) + EXTRA_CAT_COLUMNS
        if column in fit_features.columns
    ]
    model = CatBoostRegressor(**REGRESSOR_PARAMS, random_seed=SEED)
    started = time.perf_counter()
    model.fit(
        fit_features,
        fit_rows[TARGET].to_numpy(np.float64),
        cat_features=categorical,
    )
    prediction = np.clip(model.predict(audit_features), 0.001, 0.999).astype(np.float64)
    model.save_model(model_path)
    detail = {
        "fit_rows": int(len(fit_rows)),
        "audit_rows": int(len(audit_rows)),
        "feature_count": int(fit_features.shape[1]),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    del model, fit_features, audit_features
    gc.collect()
    return prediction, detail


def run(
    train_csv: Path,
    v285_axes: Path,
    v288_axes: Path,
    v288_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    is_f = train["game_type"].astype(str).eq("F")
    early = train.loc[
        train["season"].eq(2023) & train["game_month"].lt(8) & is_f
    ].reset_index(drop=True)
    late_f = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8) & is_f
    ].reset_index(drop=True)
    fit23 = train.loc[train["season"].eq(2023) & is_f].reset_index(drop=True)
    full_f24 = train.loc[train["season"].eq(2024) & is_f].reset_index(drop=True)
    late_all = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    full24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)

    reg_source, source_fit = fit_predict(
        early, late_f, output_dir / "source_brier_regressor.cbm"
    )
    reg_locked, locked_fit = fit_predict(
        fit23, full_f24, output_dir / "transfer_brier_regressor.cbm"
    )
    cls_source = np.load(
        v288_dir / "expert_late_2023_multiseed.npy", allow_pickle=False
    ).astype(np.float64)
    cls_locked = np.load(
        v288_dir / "expert_full_2024_multiseed.npy", allow_pickle=False
    ).astype(np.float64)
    if len(cls_source) != len(reg_source) or len(cls_locked) != len(reg_locked):
        raise ValueError("classifier/regressor expert alignment mismatch")

    with np.load(v285_axes, allow_pickle=False) as saved:
        parent_late = saved["baseline_late_2023"].astype(np.float64)
        parent_2024 = saved["baseline_full_2024"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        incumbent_late = saved["candidate_late_2023"].astype(np.float64)
        incumbent_2024 = saved["candidate_full_2024"].astype(np.float64)

    expert_sets = {
        "regressor": (reg_source, reg_locked),
        "classifier_regressor_mean": (
            0.5 * (cls_source + reg_source),
            0.5 * (cls_locked + reg_locked),
        ),
    }
    rows = []
    details: dict[str, Any] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    for key, (source_expert, locked_expert) in expert_sets.items():
        source_candidate = compose(parent_late, incumbent_late, late_all, source_expert)
        locked_candidate = compose(parent_2024, incumbent_2024, full24, locked_expert)
        candidates[(key, "late_2023")] = source_candidate
        candidates[(key, "full_2024")] = locked_candidate
        source_metrics = paired_metrics(late_all, incumbent_late, source_candidate)
        locked_metrics = paired_metrics(full24, incumbent_2024, locked_candidate)
        rows.append({
            "key": key,
            "source_gain": source_metrics["gain"],
            "locked_gain": locked_metrics["gain"],
        })
        details[key] = {
            "source": source_metrics,
            "locked": locked_metrics,
            "source_expert_bss": bss(late_f[TARGET].to_numpy(), source_expert),
            "locked_expert_bss": bss(full_f24[TARGET].to_numpy(), locked_expert),
        }
    selected = max(rows, key=lambda row: row["source_gain"])
    key = str(selected["key"])
    source_pass = bool(selected["source_gain"] > 0.0)
    locked_pass = bool(
        selected["locked_gain"] > 0.0
        and details[key]["locked"]["positive_month_fraction"] >= 0.625
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "multiseed_candidate" if source_pass and locked_pass else "screen_reject",
        "seed": SEED,
        "regressor_params": REGRESSOR_PARAMS,
        "futures_weight_fixed": FUTURES_WEIGHT,
        "selected": selected,
        "selected_details": details[key],
        "all_candidates": rows,
        "all_details": details,
        "fit_details": {"source": source_fit, "locked": locked_fit},
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_multiseed_confirmation": bool(source_pass and locked_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "official_train_only": True,
            "same_recent_f_windows_as_v287": True,
            "brier_aligned_rmse_objective": True,
            "v290_futures_dose_fixed": True,
            "source_only_expert_choice": True,
            "full_2024_locked_confirmation": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.save(output_dir / "expert_late_2023.npy", reg_source.astype(np.float32))
    np.save(output_dir / "expert_full_2024.npy", reg_locked.astype(np.float32))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        incumbent_late_2023=incumbent_late,
        candidate_late_2023=candidates[(key, "late_2023")],
        incumbent_full_2024=incumbent_2024,
        candidate_full_2024=candidates[(key, "full_2024")],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-axes", type=Path, required=True)
    parser.add_argument("--v288-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v285_axes, args.v288_axes, args.v288_dir, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
