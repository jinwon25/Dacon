"""Add strictly-prior count-conditioned TrackMan context to the v290 F expert.

v257 showed that these target-free arsenal/command features were not useful as
a replacement regular-season XGB.  The deployed recent-F CatBoost, however,
uses only 17k--26k fitting rows and no TrackMan context.  This experiment keeps
the v287 recipe and the deployed 10% dose fixed, appends v257's strictly-prior
features, and replaces the existing F expert.  Early-2023 -> late-2023 is the
source contract; full-2024 is locked confirmation.
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
from catboost import CatBoostClassifier

from src.archive.v287_recent_futures_direct_expert import CAT_COLUMNS, PARAMS, TARGET, build_features
from src.archive.v301_multidomain_futures_expert import bss, compose, paired_metrics


PROTOCOL = "V302_FUTURES_COUNT_TRACKMAN_EXPERT_V1"
SEED = 3020
FUTURES_WEIGHT = 0.10
EXTRA_CAT_COLUMNS = ["count_code", "same_hand", "pressure_code"]


def attach_trackman(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(supplement):
        raise ValueError("TrackMan supplement alignment mismatch")
    overlap = set(base.columns) & set(supplement.columns)
    if overlap:
        raise ValueError(f"TrackMan columns overlap base features: {sorted(overlap)}")
    return pd.concat(
        [base.reset_index(drop=True), supplement.reset_index(drop=True)], axis=1
    )


def fit_predict(
    fit_rows: pd.DataFrame,
    fit_supplement: pd.DataFrame,
    audit_rows: pd.DataFrame,
    audit_supplement: pd.DataFrame,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_features = attach_trackman(build_features(fit_rows), fit_supplement)
    audit_features = attach_trackman(build_features(audit_rows), audit_supplement).reindex(
        columns=fit_features.columns
    )
    categorical = [
        column for column in list(CAT_COLUMNS) + EXTRA_CAT_COLUMNS
        if column in fit_features.columns
    ]
    model = CatBoostClassifier(**PARAMS, random_seed=SEED)
    started = time.perf_counter()
    model.fit(
        fit_features,
        fit_rows[TARGET].to_numpy(np.int8),
        cat_features=categorical,
    )
    prediction = model.predict_proba(audit_features)[:, 1].astype(np.float64)
    model.save_model(model_path)
    detail = {
        "fit_rows": int(len(fit_rows)),
        "audit_rows": int(len(audit_rows)),
        "base_feature_count": int(len(build_features(fit_rows.head(1)).columns)),
        "trackman_feature_count": int(fit_supplement.shape[1]),
        "total_feature_count": int(fit_features.shape[1]),
        "trackman_coverage_fit": float(fit_supplement["tm_count_profile_covered"].mean()),
        "trackman_coverage_audit": float(audit_supplement["tm_count_profile_covered"].mean()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    del model, fit_features, audit_features
    gc.collect()
    return prediction, detail


def run(
    train_csv: Path,
    supplement_parquet: Path,
    v285_axes: Path,
    v288_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    supplement = pd.read_parquet(supplement_parquet)
    if len(train) != len(supplement):
        raise ValueError("full TrackMan cache alignment mismatch")
    is_f = train["game_type"].astype(str).eq("F")
    early_mask = train["season"].eq(2023) & train["game_month"].lt(8) & is_f
    late_mask = train["season"].eq(2023) & train["game_month"].ge(8) & is_f
    fit23_mask = train["season"].eq(2023) & is_f
    f24_mask = train["season"].eq(2024) & is_f
    late_all_mask = train["season"].eq(2023) & train["game_month"].ge(8)
    full24_mask = train["season"].eq(2024)

    early = train.loc[early_mask].reset_index(drop=True)
    late_f = train.loc[late_mask].reset_index(drop=True)
    fit23 = train.loc[fit23_mask].reset_index(drop=True)
    full_f24 = train.loc[f24_mask].reset_index(drop=True)
    late_all = train.loc[late_all_mask].reset_index(drop=True)
    full24 = train.loc[full24_mask].reset_index(drop=True)

    source_prediction, source_fit = fit_predict(
        early,
        supplement.loc[early_mask].reset_index(drop=True),
        late_f,
        supplement.loc[late_mask].reset_index(drop=True),
        output_dir / "source_count_trackman_f.cbm",
    )
    transfer_prediction, transfer_fit = fit_predict(
        fit23,
        supplement.loc[fit23_mask].reset_index(drop=True),
        full_f24,
        supplement.loc[f24_mask].reset_index(drop=True),
        output_dir / "transfer_count_trackman_f.cbm",
    )
    np.save(output_dir / "expert_late_2023.npy", source_prediction.astype(np.float32))
    np.save(output_dir / "expert_full_2024.npy", transfer_prediction.astype(np.float32))

    with np.load(v285_axes, allow_pickle=False) as saved:
        parent_late = saved["baseline_late_2023"].astype(np.float64)
        parent_2024 = saved["baseline_full_2024"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        incumbent_late = saved["candidate_late_2023"].astype(np.float64)
        incumbent_2024 = saved["candidate_full_2024"].astype(np.float64)
    candidate_late = compose(parent_late, incumbent_late, late_all, source_prediction)
    candidate_2024 = compose(parent_2024, incumbent_2024, full24, transfer_prediction)
    source_metrics = paired_metrics(late_all, incumbent_late, candidate_late)
    locked_metrics = paired_metrics(full24, incumbent_2024, candidate_2024)
    source_pass = bool(source_metrics["gain"] > 0.0)
    locked_pass = bool(
        locked_metrics["gain"] > 0.0
        and locked_metrics["positive_month_fraction"] >= 0.625
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "multiseed_candidate" if source_pass and locked_pass else "screen_reject",
        "seed": SEED,
        "model_params": PARAMS,
        "futures_weight_fixed": FUTURES_WEIGHT,
        "source_metrics": source_metrics,
        "locked_metrics": locked_metrics,
        "source_expert_bss": bss(late_f[TARGET].to_numpy(), source_prediction),
        "locked_expert_bss": bss(full_f24[TARGET].to_numpy(), transfer_prediction),
        "fit_details": {"source": source_fit, "locked": transfer_fit},
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_multiseed_confirmation": bool(source_pass and locked_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "official_train_and_trackman_only": True,
            "trackman_profiles_strictly_prior_season": True,
            "current_row_count_only": True,
            "v290_futures_dose_fixed": True,
            "candidate_replaces_existing_f_expert": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parent_late,
        incumbent_late_2023=incumbent_late,
        candidate_late_2023=candidate_late,
        parent_full_2024=parent_2024,
        incumbent_full_2024=incumbent_2024,
        candidate_full_2024=candidate_2024,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--supplement-parquet", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.supplement_parquet, args.v285_axes, args.v288_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
