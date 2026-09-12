"""Screen a multi-domain recent-F expert above v290.

The deployed F expert is fitted only on recent Futures rows.  This screen keeps
its feature recipe and 10% deployment dose fixed, but lets same-window regular
season rows act as a low-weight auxiliary task.  The goal is to learn player,
count, and game-state representations from a much larger sample while the
categorical game_type feature preserves the F generation process.

Auxiliary weights are selected only on early-2023 -> late-2023.  Full-2024 is
locked confirmation.  The candidate replaces, rather than stacks on top of,
the v290 F expert so the audit measures model information instead of dose.
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

from src.archive.v287_recent_futures_direct_expert import CAT_COLUMNS, PARAMS


PROTOCOL = "V301_MULTIDOMAIN_FUTURES_EXPERT_V1"
TARGET = "control_success"
FUTURES_WEIGHT = 0.10
AUXILIARY_WEIGHTS = (0.10, 0.30)
SEED = 3010


# v287 defines these outside a single exported collection.
MODEL_CAT_COLUMNS = list(CAT_COLUMNS) + ["count_code", "same_hand", "pressure_code"]
MODEL_PARAMS = dict(PARAMS)
MODEL_PARAMS.update({"iterations": 500, "thread_count": 16})


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["row_id", TARGET, "season"], errors="ignore").copy()
    for column in CAT_COLUMNS:
        output[column] = output[column].fillna("__NA__").astype(str)
    for column in output.columns.difference(CAT_COLUMNS):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(-999.0)
    output["count_code"] = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    output["same_hand"] = frame["pitcher_hand"].astype(str).eq(frame["batter_hand"].astype(str)).astype(str)
    output["pressure_code"] = (
        frame["num_runners_on"].gt(0) | frame["li"].ge(1.5)
    ).astype(str)
    return output


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    return float(100000.0 * (1.0 - np.mean((target - prediction) ** 2) / (rate * (1.0 - rate))))


def fit_predict(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    auxiliary_weight: float,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_features = build_features(fit)
    audit_features = build_features(audit).reindex(columns=fit_features.columns)
    categorical = [column for column in MODEL_CAT_COLUMNS if column in fit_features.columns]
    is_f = fit["game_type"].astype(str).eq("F").to_numpy()
    weights = np.where(is_f, 1.0, float(auxiliary_weight)).astype(np.float32)
    model = CatBoostClassifier(**MODEL_PARAMS, random_seed=SEED)
    started = time.perf_counter()
    model.fit(
        fit_features,
        fit[TARGET].to_numpy(np.int8),
        cat_features=categorical,
        sample_weight=weights,
    )
    prediction = model.predict_proba(audit_features)[:, 1].astype(np.float64)
    model.save_model(model_path)
    detail = {
        "fit_rows": int(len(fit)),
        "fit_f_rows": int(is_f.sum()),
        "fit_r_rows": int((~is_f).sum()),
        "effective_r_to_f_weight_ratio": float(auxiliary_weight),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    del model, fit_features, audit_features, weights
    gc.collect()
    return prediction, detail


def paired_metrics(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    rows = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        rows.append({
            "month": int(month),
            "gain": bss(target[mask], candidate[mask]) - bss(target[mask], baseline[mask]),
        })
    shift = candidate - baseline
    return {
        "gain": bss(target, candidate) - bss(target, baseline),
        "positive_month_fraction": float(np.mean([row["gain"] > 0.0 for row in rows])),
        "worst_month_gain": float(min(row["gain"] for row in rows)),
        "mean_abs_shift": float(np.mean(np.abs(shift))),
        "rms_shift": float(np.sqrt(np.mean(shift ** 2))),
        "changed_rows": int(np.sum(np.abs(shift) > 0.0)),
        "months": rows,
    }


def compose(
    parent: np.ndarray,
    incumbent: np.ndarray,
    frame: pd.DataFrame,
    expert: np.ndarray,
) -> np.ndarray:
    selected = frame["game_type"].astype(str).eq("F").to_numpy()
    if len(expert) != int(selected.sum()):
        raise ValueError("F expert alignment mismatch")
    output = incumbent.copy()
    output[selected] = np.clip(
        parent[selected] + FUTURES_WEIGHT * (expert - parent[selected]),
        0.001,
        0.999,
    )
    return output


def run(
    train_csv: Path,
    v285_axes: Path,
    v288_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    is_f = train["game_type"].astype(str).eq("F")

    early_2023 = train.loc[train["season"].eq(2023) & train["game_month"].lt(8)].reset_index(drop=True)
    late_2023 = train.loc[train["season"].eq(2023) & train["game_month"].ge(8)].reset_index(drop=True)
    late_f_2023 = late_2023.loc[late_2023["game_type"].astype(str).eq("F")].reset_index(drop=True)
    all_2023 = train.loc[train["season"].eq(2023)].reset_index(drop=True)
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    full_f_2024 = full_2024.loc[full_2024["game_type"].astype(str).eq("F")].reset_index(drop=True)

    with np.load(v285_axes, allow_pickle=False) as saved:
        parent_late = saved["baseline_late_2023"].astype(np.float64)
        parent_2024 = saved["baseline_full_2024"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        incumbent_late = saved["candidate_late_2023"].astype(np.float64)
        incumbent_2024 = saved["candidate_full_2024"].astype(np.float64)
    if len(parent_late) != len(late_2023) or len(incumbent_2024) != len(full_2024):
        raise ValueError("saved-axis alignment mismatch")

    rows = []
    details: dict[str, Any] = {}
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for auxiliary_weight in AUXILIARY_WEIGHTS:
        key = f"r{auxiliary_weight:g}"
        source_prediction, source_fit = fit_predict(
            early_2023,
            late_f_2023,
            auxiliary_weight,
            output_dir / f"source_{key}.cbm",
        )
        transfer_prediction, transfer_fit = fit_predict(
            all_2023,
            full_f_2024,
            auxiliary_weight,
            output_dir / f"transfer_{key}.cbm",
        )
        np.save(output_dir / f"expert_late_2023_{key}.npy", source_prediction.astype(np.float32))
        np.save(output_dir / f"expert_full_2024_{key}.npy", transfer_prediction.astype(np.float32))
        predictions[("late_2023", auxiliary_weight)] = source_prediction
        predictions[("full_2024", auxiliary_weight)] = transfer_prediction
        candidate_late = compose(parent_late, incumbent_late, late_2023, source_prediction)
        candidate_2024 = compose(parent_2024, incumbent_2024, full_2024, transfer_prediction)
        source_metrics = paired_metrics(late_2023, incumbent_late, candidate_late)
        locked_metrics = paired_metrics(full_2024, incumbent_2024, candidate_2024)
        rows.append({
            "key": key,
            "auxiliary_weight": auxiliary_weight,
            "source_gain": source_metrics["gain"],
            "locked_gain": locked_metrics["gain"],
        })
        details[key] = {
            "source": source_metrics,
            "locked": locked_metrics,
            "source_expert_bss": bss(late_f_2023[TARGET].to_numpy(), source_prediction),
            "locked_expert_bss": bss(full_f_2024[TARGET].to_numpy(), transfer_prediction),
            "source_fit": source_fit,
            "transfer_fit": transfer_fit,
        }

    selected = max(rows, key=lambda row: (row["source_gain"], -row["auxiliary_weight"]))
    selected_key = str(selected["key"])
    source_pass = bool(selected["source_gain"] > 0.0)
    locked_pass = bool(
        selected["locked_gain"] > 0.0
        and details[selected_key]["locked"]["positive_month_fraction"] >= 0.625
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "multiseed_candidate" if source_pass and locked_pass else "screen_reject",
        "model_params": MODEL_PARAMS,
        "futures_weight_fixed": FUTURES_WEIGHT,
        "selected": selected,
        "selected_details": details[selected_key],
        "all_candidates": rows,
        "all_details": details,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_multiseed_confirmation": bool(source_pass and locked_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "official_train_only": True,
            "same_window_regular_rows_are_auxiliary_only": True,
            "v290_futures_dose_fixed": True,
            "candidate_replaces_existing_f_expert": True,
            "source_only_auxiliary_weight_selection": True,
            "full_2024_locked_confirmation": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    selected_weight = float(selected["auxiliary_weight"])
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parent_late,
        incumbent_late_2023=incumbent_late,
        candidate_late_2023=compose(parent_late, incumbent_late, late_2023, predictions[("late_2023", selected_weight)]),
        parent_full_2024=parent_2024,
        incumbent_full_2024=incumbent_2024,
        candidate_full_2024=compose(parent_2024, incumbent_2024, full_2024, predictions[("full_2024", selected_weight)]),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v285_axes, args.v288_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
