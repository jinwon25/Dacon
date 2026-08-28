"""Cross-fit a row-local residual calibrator above the exact current contract.

Protocol
--------
1. Fit small CatBoost regressors to ``target - current_probability`` on the
   exact 2022 forward rows only.
2. Select one fixed recipe and correction dose on the untouched late-2023
   forward axis.
3. Refit that frozen recipe on 2022 + late-2023 and open full-2024 once.

Only official per-row pre-pitch columns and the row's current prediction are
features.  No evaluation-row aggregate, ordering operation, or target-derived
test feature exists in the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.archive.v168_jy_exact_contract_reaudit import (
    C3_BASE_RECENT_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    c3_mix,
    compose,
    metrics,
    rcore_mask,
)
from src.core.contract import _load_contract_axis
from src.data import read_main


PROTOCOL = "V170_CURRENT_RESIDUAL_CATBOOST_V1"
RECIPES = (
    {"name": "depth4_l2_100", "depth": 4, "l2_leaf_reg": 100.0},
    {"name": "depth6_l2_200", "depth": 6, "l2_leaf_reg": 200.0},
)
DOSES = (0.25, 0.50, 0.75, 1.00)
CORRECTION_CLIP = 0.05


def feature_frame(frame: pd.DataFrame, base: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    excluded = {"row_id", "control_success", "season"}
    output = frame.drop(
        columns=[column for column in excluded if column in frame], errors="ignore"
    ).copy()
    output["base_prediction"] = np.asarray(base, dtype=np.float64)
    output["count_state"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    output["platoon"] = (
        frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    )
    output["pitcher_count"] = (
        frame["pitcher_id"].astype(str) + "-" + output["count_state"]
    )
    output["pitcher_batter_hand"] = (
        frame["pitcher_id"].astype(str) + "-" + frame["batter_hand"].astype(str)
    )
    output["batter_count"] = (
        frame["batter_id"].astype(str) + "-" + output["count_state"]
    )
    output["team_matchup"] = (
        frame["pitcher_team_id"].astype(str)
        + "-"
        + frame["batter_team_id"].astype(str)
    )
    output["situation_state"] = (
        output["count_state"]
        + "-"
        + frame["outs_before"].astype(str)
        + "-"
        + frame["base_state"].astype(str)
    )
    output["pitcher_history_log_n"] = np.log1p(
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0)
    )
    output["batter_history_log_n"] = np.log1p(
        pd.to_numeric(frame["asof_batter_n"], errors="coerce").fillna(0.0)
    )
    output["pitchmix_history_log_n"] = np.log1p(
        pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0)
    )
    output["recent_success_1_minus_5"] = (
        pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(frame["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    output["recent_middle_1_minus_5"] = (
        pd.to_numeric(frame["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
        - pd.to_numeric(frame["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    )
    output["pitcher_minus_batter_success"] = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        - pd.to_numeric(frame["asof_batter_success_rate"], errors="coerce")
    )
    categorical = [
        "game_dayofweek",
        "top_bottom",
        "game_type",
        "base_state",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "count_state",
        "platoon",
        "pitcher_count",
        "pitcher_batter_hand",
        "batter_count",
        "team_matchup",
        "situation_state",
    ]
    categorical = [column for column in categorical if column in output]
    for column in categorical:
        output[column] = output[column].astype("string").fillna("__MISSING__").astype(str)
    for column in output.columns:
        if column not in categorical:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(
                "float32"
            )
    return output, categorical


def make_model(recipe: dict[str, Any], seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=400,
        depth=int(recipe["depth"]),
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=float(recipe["l2_leaf_reg"]),
        random_strength=0.5,
        random_seed=int(seed),
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        one_hot_max_size=32,
    )


def apply_correction(
    base: np.ndarray,
    correction: np.ndarray,
    active: np.ndarray,
    dose: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    bounded = np.clip(np.asarray(correction, dtype=np.float64), -CORRECTION_CLIP, CORRECTION_CLIP)
    output[active] = np.clip(
        output[active] + float(dose) * bounded[active], 0.001, 0.999
    )
    return output


def _build_contracts(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, float],
]:
    train = read_main(train_csv)
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    bases = {}
    for name, values in source.items():
        bases[name] = compose(
            values["component"],
            values["h1"],
            c3_mix(values["sign"], values["recent"], C3_BASE_RECENT_WEIGHT),
            axes[name],
            h1_weight=H1_BASE_WEIGHT,
        )
    locked, parity = _locked_contract(
        axes["full_2024"],
        frames[2024],
        correction[2024],
        h1_path,
        c3_path,
        v160_path,
    )
    bases["full_2024"] = locked["current"]
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    del late23
    return train, axis_frames, axes, bases, parity


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _train, frames, axes, bases, parity = _build_contracts(
        train_csv,
        contract_dir,
        v104_path,
        h1_path,
        c3_path,
        v160_path,
        bridge_oof,
    )
    features = {}
    categorical = None
    for name in axes:
        features[name], current_categorical = feature_frame(frames[name], bases[name])
        if categorical is None:
            categorical = current_categorical
        elif categorical != current_categorical:
            raise ValueError("categorical feature mismatch across axes")
    assert categorical is not None

    fit_mask = axes["full_2022"]["exact_mask"].astype(bool)
    validation_mask = axes["late_2023"]["exact_mask"].astype(bool)
    locked_mask = axes["full_2024"]["exact_mask"].astype(bool)
    target_residual_22 = axes["full_2022"]["target"] - bases["full_2022"]
    trials = []
    fitted = {}
    validation_corrections = {}
    for recipe_index, recipe in enumerate(RECIPES):
        print(f"[v170] fitting selection model {recipe['name']}", flush=True)
        model = make_model(recipe, seed=17042 + recipe_index)
        started = time.perf_counter()
        model.fit(
            features["full_2022"].loc[fit_mask],
            target_residual_22[fit_mask],
            cat_features=categorical,
        )
        elapsed = time.perf_counter() - started
        correction = np.asarray(model.predict(features["late_2023"]), dtype=np.float64)
        validation_corrections[recipe["name"]] = correction
        fitted[recipe["name"]] = model
        for dose in DOSES:
            active = rcore_mask(axes["late_2023"])
            candidate = apply_correction(
                bases["late_2023"], correction, active, dose
            )
            result = metrics(axes["late_2023"], bases["late_2023"], candidate)
            trials.append(
                {
                    "recipe": recipe["name"],
                    "dose": dose,
                    "fit_seconds": elapsed,
                    "validation": result,
                }
            )
    eligible = [
        row
        for row in trials
        if row["validation"]["gain"] > 0.0
        and row["validation"]["positive_month_fraction"] >= 2.0 / 3.0
    ]
    selected = max(
        eligible if eligible else trials,
        key=lambda row: (
            row["validation"]["gain"],
            row["validation"]["worst_month_gain"],
            -float(row["dose"]),
        ),
    )
    selected_recipe = next(
        recipe for recipe in RECIPES if recipe["name"] == selected["recipe"]
    )

    # Honest pre-refit transfer: the model that saw 2022 only is also scored on
    # 2024 before the late-2023 labels are added.
    selection_model = fitted[selected["recipe"]]
    correction_24_pre_refit = np.asarray(
        selection_model.predict(features["full_2024"]), dtype=np.float64
    )
    candidate_24_pre_refit = apply_correction(
        bases["full_2024"],
        correction_24_pre_refit,
        rcore_mask(axes["full_2024"]),
        float(selected["dose"]),
    )
    locked_pre_refit = metrics(
        axes["full_2024"], bases["full_2024"], candidate_24_pre_refit
    )

    # Refit the frozen recipe using the now-open validation origin, then make
    # the protocol's primary full-2024 locked prediction.
    print(
        f"[v170] refitting frozen {selected['recipe']} dose={selected['dose']}",
        flush=True,
    )
    combined_features = pd.concat(
        [
            features["full_2022"].loc[fit_mask],
            features["late_2023"].loc[validation_mask],
        ],
        ignore_index=True,
    )
    combined_residual = np.concatenate(
        [
            target_residual_22[fit_mask],
            (axes["late_2023"]["target"] - bases["late_2023"])[validation_mask],
        ]
    )
    refit = make_model(selected_recipe, seed=17142)
    started = time.perf_counter()
    refit.fit(combined_features, combined_residual, cat_features=categorical)
    refit_seconds = time.perf_counter() - started
    correction_24 = np.asarray(refit.predict(features["full_2024"]), dtype=np.float64)
    candidate_24 = apply_correction(
        bases["full_2024"],
        correction_24,
        rcore_mask(axes["full_2024"]),
        float(selected["dose"]),
    )
    locked = metrics(axes["full_2024"], bases["full_2024"], candidate_24)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selection_model, output_dir / "selection_model.joblib")
    joblib.dump(refit, output_dir / "refit_model.joblib")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        current_full_2024=bases["full_2024"],
        correction_full_2024=correction_24,
        candidate_full_2024=candidate_24,
        candidate_full_2024_pre_refit=candidate_24_pre_refit,
    )
    result = {
        "protocol": PROTOCOL,
        "status": (
            "locked_audit_complete"
            if eligible
            else "locked_reject_no_source_candidate"
        ),
        "current_contract_parity": parity,
        "recipes": list(RECIPES),
        "doses": list(DOSES),
        "correction_clip": CORRECTION_CLIP,
        "selection_trials": trials,
        "source_gate_passed": bool(eligible),
        "selected_on_late_2023": selected,
        "selected_is_diagnostic_fallback": not bool(eligible),
        "locked_2024_pre_refit": locked_pre_refit,
        "locked_2024_refit": locked,
        "refit_seconds": refit_seconds,
        "eligible_for_packaging": False,
        "limitations": [
            "Full 2024 has been repeatedly used by the project and remains development-contaminated.",
            "A final deployable refit would need exact current-contract OOF for every included training origin.",
        ],
        "restrictions": {
            "official_train_only": True,
            "external_2025_outcomes_used": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "locked_2024_pre_refit": locked_pre_refit,
        "locked_2024_refit": locked,
    }, ensure_ascii=False, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
