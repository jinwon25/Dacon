"""Audit an ID-free LightGBM population expert above the v290 F specialist.

The deployed v290 F model is an ID-rich CatBoost expert.  This experiment fits
an independently parameterized, one-hot population LightGBM that deliberately
excludes pitcher and batter IDs.  Three combination rules are frozen: replace
half of the existing expert at the same total dose, add the population expert
only where both experts move the v244 parent in the same direction, or add it
on every F row.  Rule selection uses early-to-late 2023 only and full-2024 is
opened once as confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V293_FUTURES_POPULATION_LGB_DIVERSITY_V1"
TARGET = "control_success"
F_WEIGHT = 0.10
DROP_COLUMNS = ["row_id", TARGET, "season", "pitcher_id", "batter_id", "game_type"]
CAT_COLUMNS = [
    "top_bottom",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 120,
    "max_bin": 127,
    "reg_alpha": 1.0,
    "reg_lambda": 8.0,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "objective": "binary",
    "verbosity": -1,
    "n_jobs": 16,
}


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    output["count_code"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    output["same_hand"] = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).astype(str)
    output["pressure_code"] = (
        frame["num_runners_on"].gt(0) | frame["li"].ge(1.5)
    ).astype(str)
    categorical = CAT_COLUMNS + ["count_code", "same_hand", "pressure_code"]
    for column in categorical:
        output[column] = output[column].fillna("__NA__").astype(str)
    for column in output.columns.difference(categorical):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _pipeline(features: pd.DataFrame, seed: int) -> Pipeline:
    categorical = [
        column
        for column in CAT_COLUMNS + ["count_code", "same_hand", "pressure_code"]
        if column in features.columns
    ]
    numeric = [column for column in features.columns if column not in categorical]
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )
    model = LGBMClassifier(**PARAMS, random_state=int(seed))
    return Pipeline([("pre", pre), ("model", model)])


def _fit_predict(
    fit_rows: pd.DataFrame, audit_rows: pd.DataFrame, seed: int
) -> tuple[np.ndarray, Pipeline]:
    fit_features = build_features(fit_rows)
    audit_features = build_features(audit_rows).reindex(columns=fit_features.columns)
    model = _pipeline(fit_features, seed)
    model.fit(fit_features, fit_rows[TARGET].to_numpy(np.int8))
    return model.predict_proba(audit_features)[:, 1].astype(np.float64), model


def _axis(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "target": frame[TARGET].to_numpy(np.float64),
        "exact_mask": np.ones(len(frame), dtype=bool),
        "game_month": frame["game_month"].to_numpy(),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
    }


def _compose(
    rule: str,
    base: np.ndarray,
    v290_parent: np.ndarray,
    f_mask: np.ndarray,
    cat_expert: np.ndarray,
    lgb_expert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(v290_parent, dtype=np.float64).copy()
    cat_move = cat_expert - base[f_mask]
    lgb_move = lgb_expert - base[f_mask]
    if rule == "same_total_mean":
        output[f_mask] = np.clip(
            base[f_mask] + F_WEIGHT * (0.5 * cat_move + 0.5 * lgb_move),
            0.001,
            0.999,
        )
        active = f_mask
    elif rule == "consensus_add":
        consensus_f = cat_move * lgb_move > 0.0
        active = np.zeros(len(base), dtype=bool)
        active[np.flatnonzero(f_mask)[consensus_f]] = True
        output[active] = np.clip(
            output[active] + F_WEIGHT * lgb_move[consensus_f], 0.001, 0.999
        )
    elif rule == "full_add":
        output[f_mask] = np.clip(
            output[f_mask] + F_WEIGHT * lgb_move, 0.001, 0.999
        )
        active = f_mask
    else:
        raise ValueError(f"unknown rule: {rule}")
    return output, active


def run(
    train_csv: Path,
    exact_axes: Path,
    v290_axes: Path,
    cat_expert_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    is_f = train["game_type"].astype(str).eq("F")
    early_2023 = train.loc[
        train["season"].eq(2023) & is_f & train["game_month"].lt(8)
    ].reset_index(drop=True)
    late_f_2023 = train.loc[
        train["season"].eq(2023) & is_f & train["game_month"].ge(8)
    ].reset_index(drop=True)
    full_f_2023 = train.loc[train["season"].eq(2023) & is_f].reset_index(drop=True)
    late_all_2023 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    full_f_2024 = full_2024.loc[
        full_2024["game_type"].astype(str).eq("F")
    ].reset_index(drop=True)

    lgb_source, source_model = _fit_predict(early_2023, late_f_2023, seed)
    lgb_locked, locked_model = _fit_predict(full_f_2023, full_f_2024, seed + 1)
    joblib.dump(source_model, output_dir / "early23_to_late23_population_lgb.joblib", compress=3)
    joblib.dump(locked_model, output_dir / "full23_to_full24_population_lgb.joblib", compress=3)
    np.save(output_dir / "expert_late_2023.npy", lgb_source.astype(np.float32))
    np.save(output_dir / "expert_full_2024.npy", lgb_locked.astype(np.float32))

    with np.load(exact_axes, allow_pickle=False) as saved:
        base = {
            "late_2023": saved["baseline_late_2023"].astype(np.float64),
            "full_2024": saved["baseline_full_2024"].astype(np.float64),
        }
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent = {
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        }
    frames = {"late_2023": late_all_2023, "full_2024": full_2024}
    f_masks = {
        axis: frame["game_type"].astype(str).eq("F").to_numpy()
        for axis, frame in frames.items()
    }
    cat = {
        "late_2023": np.load(
            cat_expert_dir / "expert_late_2023_multiseed.npy", allow_pickle=False
        ).astype(np.float64),
        "full_2024": np.load(
            cat_expert_dir / "expert_full_2024_multiseed.npy", allow_pickle=False
        ).astype(np.float64),
    }
    lgb = {"late_2023": lgb_source, "full_2024": lgb_locked}

    rules = ("same_total_mean", "consensus_add", "full_add")
    candidates: dict[tuple[str, str], np.ndarray] = {}
    actives: dict[tuple[str, str], np.ndarray] = {}
    details: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for rule in rules:
        per_axis: dict[str, Any] = {}
        for axis in frames:
            candidate, active = _compose(
                rule, base[axis], parent[axis], f_masks[axis], cat[axis], lgb[axis]
            )
            candidates[(rule, axis)] = candidate
            actives[(rule, axis)] = active
            per_axis[axis] = paired_metrics(
                _axis(frames[axis]), parent[axis], candidate, active
            )
        details[rule] = per_axis
        rows.append(
            {
                "rule": rule,
                "source_gain_vs_v290": per_axis["late_2023"]["gain"],
                "locked_gain_vs_v290": per_axis["full_2024"]["gain"],
            }
        )

    # The 2023 source chooses the rule once; 2024 is not consulted.
    selected = max(rows, key=lambda row: row["source_gain_vs_v290"])
    selected_rule = str(selected["rule"])
    robustness = _robustness(
        _axis(full_2024),
        parent["full_2024"],
        candidates[(selected_rule, "full_2024")],
        actives[(selected_rule, "full_2024")],
        [candidates[(rule, "full_2024")] for rule in rules],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    promote = bool(
        selected["source_gain_vs_v290"] > 0.0
        and selected["locked_gain_vs_v290"] > 0.0
        and robust_pass
    )
    corr = {
        axis: float(np.corrcoef(cat[axis], lgb[axis])[0, 1]) for axis in frames
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "numeric_promote" if promote else "rejected",
        "f_weight": F_WEIGHT,
        "seed": int(seed),
        "model_params": PARAMS,
        "id_columns_excluded": ["pitcher_id", "batter_id"],
        "cat_lgb_prediction_correlation": corr,
        "selected": selected,
        "all_rules": rows,
        "selected_metrics": details[selected_rule],
        "locked_robustness": robustness,
        "numeric_promotion_gate_passed": promote,
        "restrictions": {
            "official_train_only": True,
            "independent_id_free_model_family": True,
            "fixed_candidate_rules": True,
            "rule_selected_on_late23_only": True,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_late_2023=parent["late_2023"],
        candidate_late_2023=candidates[(selected_rule, "late_2023")],
        active_late_2023=actives[(selected_rule, "late_2023")],
        parent_full_2024=parent["full_2024"],
        candidate_full_2024=candidates[(selected_rule, "full_2024")],
        active_full_2024=actives[(selected_rule, "full_2024")],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--cat-expert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2930)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.exact_axes,
                args.v290_axes,
                args.cat_expert_dir,
                args.output_dir,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
