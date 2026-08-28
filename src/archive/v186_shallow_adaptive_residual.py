"""Fit a shallow, identity-free adaptive residual above JY plus v178.

Model structure and dose are selected jointly on two forward source checks:
early-2022 to late-2022 and full-2022 to late-2023.  The selected recipe is
then refit on source residuals and evaluated once on the 2024 locked contract.
Only official labelled rows and row-local features are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v178_jy_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V186_SHALLOW_ADAPTIVE_RESIDUAL_V1"
V178_SCALE = 0.25
CORRECTION_CAP = 0.025
ETA_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)
MODEL_CONFIGS = (
    {"name": "d2_i73_l30", "depth": 2, "iterations": 73, "l2": 30.0},
    {"name": "d2_i120_l100", "depth": 2, "iterations": 120, "l2": 100.0},
    {"name": "d3_i73_l30", "depth": 3, "iterations": 73, "l2": 30.0},
    {"name": "d3_i120_l100", "depth": 3, "iterations": 120, "l2": 100.0},
)
CATEGORICAL = ("top_bottom", "game_type", "base_state", "count_state", "hand_matchup")
RAW_NUMERIC = (
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "home_win_expectancy",
    "away_win_expectancy", "li", "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_pitchmix_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in axis.items()}


def meta_features(
    frame: pd.DataFrame,
    parent: np.ndarray,
    v178: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    output = pd.DataFrame(index=frame.index)
    output["parent_prediction"] = np.asarray(parent, dtype=np.float32)
    output["v178_prediction"] = np.asarray(v178, dtype=np.float32)
    output["v178_shift"] = np.asarray(v178 - parent, dtype=np.float32)
    for column in RAW_NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        if column in {"asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"}:
            values = np.log1p(values.clip(lower=0.0))
        output[column] = values.astype(np.float32)

    success_recent = frame[
        [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
    ].apply(pd.to_numeric, errors="coerce")
    middle_recent = frame[
        [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
    ].apply(pd.to_numeric, errors="coerce")
    output["recent_success_mean"] = success_recent.mean(axis=1).astype(np.float32)
    output["recent_success_std"] = success_recent.std(axis=1).astype(np.float32)
    output["recent_success_gap"] = (
        success_recent.mean(axis=1)
        - pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
    ).astype(np.float32)
    output["recent_success_trend"] = (
        success_recent.iloc[:, 0] - success_recent.iloc[:, 2]
    ).astype(np.float32)
    output["recent_middle_mean"] = middle_recent.mean(axis=1).astype(np.float32)
    output["recent_middle_std"] = middle_recent.std(axis=1).astype(np.float32)
    output["failure_profile_sum"] = (
        pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
        + pd.to_numeric(frame["asof_pitcher_reverse_rate"], errors="coerce")
    ).astype(np.float32)

    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    output["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    output["hand_matchup"] = (
        frame["pitcher_hand"].astype("string").fillna("NA")
        + "-"
        + frame["batter_hand"].astype("string").fillna("NA")
    )
    for column in ("top_bottom", "game_type", "base_state"):
        output[column] = frame[column]
    categorical = [column for column in CATEGORICAL if column in output]
    for column in categorical:
        output[column] = output[column].astype("string").fillna("NA").astype(str)
    feature_names = output.columns.tolist()
    numeric = [column for column in feature_names if column not in categorical]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return output, feature_names, categorical


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    query_features: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    config: dict[str, Any],
    *,
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, CatBoostRegressor]:
    model = CatBoostRegressor(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=0.025,
        loss_function="RMSE",
        l2_leaf_reg=float(config["l2"]),
        random_strength=0.2,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=186000 + int(seed_offset),
        thread_count=16,
        allow_writing_files=False,
        verbose=False,
    )
    fit_pool = Pool(
        fit_features[feature_names], label=np.asarray(fit_target, dtype=np.float64),
        weight=sample_weight, cat_features=categorical,
    )
    query_pool = Pool(query_features[feature_names], cat_features=categorical)
    model.fit(fit_pool)
    correction = np.clip(model.predict(query_pool), -CORRECTION_CAP, CORRECTION_CAP)
    return correction.astype(np.float64), model


def apply_correction(
    base: np.ndarray,
    correction: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(correction)[active], 0.001, 0.999
    )
    return output


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, post4 = _load_year_context(train_csv)
    full_train = pd.read_csv(train_csv, low_memory=False)
    full_season = full_train["season"].to_numpy(np.int16)
    feature_year_frames = {
        year: full_train.loc[full_season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23_mask = raw_frames[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": feature_year_frames[2022],
        "late_2023": feature_year_frames[2023].loc[late23_mask].reset_index(drop=True),
        "full_2024": feature_year_frames[2024],
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen_weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], frozen_weights, library_root)
        for name in v158_base
    }
    base = {
        name: apply_direction(parents[name], directions[name], V178_SCALE)
        for name in parents
    }
    feature_frames: dict[str, pd.DataFrame] = {}
    feature_names: list[str] = []
    categorical: list[str] = []
    for name in frames:
        feature_frames[name], names, cats = meta_features(
            frames[name], parents[name], base[name]
        )
        if not feature_names:
            feature_names, categorical = names, cats
        elif names != feature_names or cats != categorical:
            raise ValueError("meta feature schema mismatch")

    exact22 = np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
    early22 = frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = ~early22
    fit_early22 = exact22 & early22
    fit_full22 = exact22
    exact23 = np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
    axis_late22 = _slice_axis(axes["full_2022"], late22)
    trials: list[dict[str, Any]] = []
    source_details: dict[str, Any] = {}
    for config in MODEL_CONFIGS:
        correction22, _ = fit_predict(
            feature_frames["full_2022"].loc[fit_early22],
            np.asarray(axes["full_2022"]["target"])[fit_early22] - base["full_2022"][fit_early22],
            feature_frames["full_2022"].loc[late22],
            feature_names, categorical, config, seed_offset=22,
        )
        correction23, _ = fit_predict(
            feature_frames["full_2022"].loc[fit_full22],
            np.asarray(axes["full_2022"]["target"])[fit_full22] - base["full_2022"][fit_full22],
            feature_frames["late_2023"],
            feature_names, categorical, config, seed_offset=23,
        )
        for eta in ETA_GRID:
            key = f"{config['name']}__e{eta:g}"
            candidate22 = apply_correction(
                base["full_2022"][late22], correction22,
                exact22[late22], eta,
            )
            candidate23 = apply_correction(
                base["late_2023"], correction23, exact23, eta,
            )
            detail = {
                "early22_to_late22": metrics(
                    axis_late22, base["full_2022"][late22], candidate22
                ),
                "full22_to_late23": metrics(
                    axes["late_2023"], base["late_2023"], candidate23
                ),
            }
            passed = all(source_gate(item) for item in detail.values())
            source_details[key] = detail
            trials.append(
                {
                    "key": key,
                    "model": config["name"],
                    "eta": eta,
                    "source_gate_passed": passed,
                    "minimum_gain": min(item["gain"] for item in detail.values()),
                    "mean_gain": float(np.mean([item["gain"] for item in detail.values()])),
                    "minimum_positive_month_fraction": min(
                        item["positive_month_fraction"] for item in detail.values()
                    ),
                    "worst_month_gain": min(item["worst_month_gain"] for item in detail.values()),
                }
            )
    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=[False, False, False, False], kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_top": ranking.head(10).to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected = passing.iloc[0]
        selected_config = next(
            config for config in MODEL_CONFIGS if config["name"] == selected["model"]
        )
        eta = float(selected["eta"])
        fit_frame = pd.concat(
            [feature_frames["full_2022"].loc[exact22], feature_frames["late_2023"].loc[exact23]],
            ignore_index=True,
        )
        fit_target = np.concatenate(
            [
                np.asarray(axes["full_2022"]["target"])[exact22] - base["full_2022"][exact22],
                np.asarray(axes["late_2023"]["target"])[exact23] - base["late_2023"][exact23],
            ]
        )
        sample_weight = np.concatenate(
            [np.full(int(exact22.sum()), 0.55), np.ones(int(exact23.sum()))]
        )
        correction24, model = fit_predict(
            fit_frame, fit_target, feature_frames["full_2024"],
            feature_names, categorical, selected_config,
            sample_weight=sample_weight, seed_offset=24,
        )
        exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        candidate24 = apply_correction(base["full_2024"], correction24, exact24, eta)
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        incremental = metrics(axes["full_2024"], base["full_2024"], candidate24)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate24,
            exact24, [candidate24, base["full_2024"]],
        )
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        model.save_model(output_dir / "selected_meta.cbm")
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], base=base["full_2024"],
            candidate=candidate24, correction=correction24, active=exact24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "selected_on_sources_only": selected.to_dict(),
            "selected_model_config": selected_config,
            "source": source_details[str(selected["key"])],
            "locked_2024": locked,
            "incremental_over_v178": incremental,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "feature_names": feature_names,
            "categorical": categorical,
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_labels_only": True,
        "strict_forward_source_checks": True,
        "pitcher_or_batter_identity_feature_used": False,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_prediction_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
        "locked_2024_development_contaminated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
