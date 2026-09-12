"""Screen prior-season pitcher-role workload features in the exact H1 model.

The row-local 1/3/5-game rates identify only a minimum compatible pitch count.
This experiment estimates a pitcher's usual appearance length using row counts
from strictly earlier seasons, then lifts each minimum denominator to a role-
compatible multiple.  No target values are used to build these features.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import affine, metrics
from src.archive.v199_recent_workload_reconstruction import (
    HORIZONS,
    MAX_DENOMINATOR,
    attach_main_game_index,
    infer_denominators,
    rate_columns,
)
from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.archive.v203_h1_workload_feature_screen import MODEL_CONFIG
from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    strict_axis,
)
from src.archive.v214_joint_workload_h1_screen import joint_workload_features
from src.champion.v130_catboost_independent_oof_blend import post4
from src.champion.v131_catboost_h1_independent_oof import _fit_year, _prepare_features


PROTOCOL = "V220_PRIOR_PITCHER_ROLE_WORKLOAD_H1_V1"
YEARS = (2022, 2023, 2024)
SCREEN_SCALE = 0.10


def _appearance_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "inning", "top_bottom", "pitcher_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing appearance columns: {sorted(missing)}")
    indexed = attach_main_game_index(
        frame[["season", "inning", "top_bottom", "pitcher_id"]]
    )
    return (
        indexed.groupby(
            ["season", "main_game_index", "pitcher_id"],
            sort=False,
            observed=True,
        )
        .size()
        .rename("pitches")
        .reset_index()
    )


def _role_stats(games: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    grouped = games.groupby("pitcher_id", sort=False, observed=True)["pitches"]
    stats = grouped.agg(["count", "mean", "median", "std", "max"])
    stats["q25"] = grouped.quantile(0.25)
    stats["q75"] = grouped.quantile(0.75)
    stats["starter_rate"] = grouped.apply(lambda value: float(np.mean(value >= 40)))
    stats["bulk_rate"] = grouped.apply(lambda value: float(np.mean(value >= 60)))
    stats["short_rate"] = grouped.apply(lambda value: float(np.mean(value <= 25)))
    stats = stats.fillna(0.0).add_prefix(prefix).reset_index()
    return stats


def prior_pitcher_role_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features for every row from that pitcher's strictly prior seasons."""

    games = _appearance_table(frame)
    output = pd.DataFrame(index=frame.index)
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(np.int16)
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise")
    stat_columns = (
        "count", "mean", "median", "std", "max", "q25", "q75",
        "starter_rate", "bulk_rate", "short_rate",
    )
    for year in sorted(np.unique(seasons)):
        row_mask = seasons == year
        row_index = frame.index[row_mask]
        prior = games.loc[games["season"] < year].copy()
        if prior.empty:
            for scope in ("last", "all"):
                for stat in stat_columns:
                    output.loc[row_index, f"role_{scope}_{stat}"] = 0.0
            output.loc[row_index, "role_prior_season_count"] = 0.0
            continue
        latest = (
            prior.groupby("pitcher_id", sort=False, observed=True)["season"]
            .max()
            .rename("latest_season")
            .reset_index()
        )
        last = prior.merge(latest, on="pitcher_id", how="inner", validate="many_to_one")
        last = last.loc[last["season"] == last["latest_season"]]
        last_stats = _role_stats(last, "role_last_").set_index("pitcher_id")
        all_stats = _role_stats(prior, "role_all_").set_index("pitcher_id")
        season_count = (
            prior.groupby("pitcher_id", sort=False, observed=True)["season"]
            .nunique()
        )
        ids = pitcher.loc[row_index]
        for stat in stat_columns:
            output.loc[row_index, f"role_last_{stat}"] = (
                ids.map(last_stats[f"role_last_{stat}"]).fillna(0.0).to_numpy()
            )
            output.loc[row_index, f"role_all_{stat}"] = (
                ids.map(all_stats[f"role_all_{stat}"]).fillna(0.0).to_numpy()
            )
        output.loc[row_index, "role_prior_season_count"] = (
            ids.map(season_count).fillna(0.0).to_numpy()
        )

    output = output.fillna(0.0)
    output["role_has_history"] = (output["role_all_count"] > 0.0).astype(float)
    output["role_last_log_apps"] = np.log1p(output["role_last_count"])
    output["role_all_log_apps"] = np.log1p(output["role_all_count"])
    output["role_last_iqr"] = output["role_last_q75"] - output["role_last_q25"]
    output["role_last_minus_all_median"] = (
        output["role_last_median"] - output["role_all_median"]
    )

    role_reference = output["role_last_median"].to_numpy(np.float64)
    has_history = output["role_has_history"].to_numpy(bool)
    career_success = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        frame["asof_pitcher_middle_rate"], errors="coerce"
    ).fillna(0.2).to_numpy(np.float64)
    priors = {1: 20.0, 3: 60.0, 5: 100.0}
    for horizon in HORIZONS:
        success_column, middle_column = rate_columns(horizon)
        success = pd.to_numeric(frame[success_column], errors="coerce").to_numpy(np.float64)
        middle = pd.to_numeric(frame[middle_column], errors="coerce").to_numpy(np.float64)
        inferred = infer_denominators(
            pd.Series(success), pd.Series(middle),
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        minimum = pd.to_numeric(
            inferred["minimum_denominator"], errors="coerce"
        ).fillna(0.0).to_numpy(np.float64)
        valid = (
            inferred["rounded_rational_fit"].fillna(False).to_numpy(bool)
            & np.isfinite(success) & np.isfinite(middle)
            & (minimum > 0.0) & has_history
        )
        multiplier = np.maximum(
            1.0,
            np.rint(np.divide(
                role_reference * horizon,
                minimum,
                out=np.ones(len(frame), dtype=np.float64),
                where=minimum > 0.0,
            )),
        )
        candidate = np.minimum(minimum * multiplier, MAX_DENOMINATOR[horizon])
        candidate = np.where(valid, candidate, 0.0)
        reliability = candidate / (candidate + priors[horizon])
        output[f"role_prev{horizon}_log_n"] = np.log1p(candidate)
        output[f"role_prev{horizon}_multiplier"] = np.where(valid, multiplier, 0.0)
        output[f"role_prev{horizon}_reliability"] = reliability
        output[f"role_prev{horizon}_success_reliable_delta"] = np.where(
            valid, reliability * (success - career_success), 0.0
        )
        output[f"role_prev{horizon}_middle_reliable_delta"] = np.where(
            valid, reliability * (middle - career_middle), 0.0
        )
    return output.astype(np.float32)


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_role_stats": True,
        "pitch_counts_from_row_counts_not_target": True,
        "fixed_role_compatible_multiple_rule": True,
        "fixed_seed42_screen": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    component_root: Path,
    baseline_checkpoint_dir: Path,
    joint_checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, component_root
    )
    workload = workload_feature_frame(train)
    joint = joint_workload_features(train)
    role = prior_pitcher_role_features(train)
    feature_names = (
        list(h1_features) + list(workload.columns) + list(joint.columns)
        + list(role.columns)
    )
    for feature_frame in (workload, joint, role):
        for column in feature_frame.columns:
            train[column] = feature_frame[column].to_numpy(np.float32)
    del workload, joint, role
    gc.collect()

    details: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, float]] = {}
    fitted: list[dict[str, Any]] = []
    for year in YEARS:
        checkpoint = output_dir / f"role_joint_h1_year{year}_seed42.npy"
        if checkpoint.exists():
            role_raw = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            reused = True
        else:
            role_raw = _fit_year(
                train, target, season, year, feature_names, MODEL_CONFIG,
                "prior-role-joint-workload-H1-seed42",
            )
            np.save(checkpoint, role_raw, allow_pickle=False)
            reused = False
        baseline_raw = np.load(
            baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        joint_raw = np.load(
            joint_checkpoint_dir / f"joint_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64)
        audit = season == year
        frame = train.loc[audit].reset_index(drop=True)
        correction = post4(train.loc[season < year].reset_index(drop=True), frame)
        baseline = affine(baseline_raw + correction)
        joint_prediction = affine(joint_raw + correction)
        role_prediction = affine(role_raw + correction)
        axis = strict_axis(frame, target[audit], baseline)
        joint_candidate = apply_component_delta(
            baseline, joint_prediction, SCREEN_SCALE
        )
        role_candidate = apply_component_delta(
            baseline, role_prediction, SCREEN_SCALE
        )
        joint_result = metrics(axis, baseline, joint_candidate)
        role_result = metrics(axis, baseline, role_candidate)
        details[str(year)] = role_result
        comparisons[str(year)] = {
            "joint_gain": float(joint_result["gain"]),
            "role_joint_gain": float(role_result["gain"]),
            "incremental_gain": float(role_result["gain"] - joint_result["gain"]),
            "raw_prediction_correlation": float(
                np.corrcoef(joint_raw, role_raw)[0, 1]
            ),
        }
        fitted.append({
            "year": year,
            "rows": len(role_raw),
            "reused": reused,
            "checkpoint": str(checkpoint),
        })

    all_positive = all(details[str(year)]["gain"] > 0.0 for year in YEARS)
    source_month_pass = all(
        details[str(year)]["positive_month_fraction"] >= 0.70
        and details[str(year)]["worst_month_gain"] > -5.0
        for year in (2022, 2023)
    )
    incremental_years = sum(
        comparisons[str(year)]["incremental_gain"] > 0.0 for year in YEARS
    )
    eligible = bool(all_positive and source_month_pass and incremental_years >= 2)
    summary = {
        "protocol": PROTOCOL,
        "status": "confirm_multiseed" if eligible else "screen_reject",
        "screen_scale": SCREEN_SCALE,
        "model_config": MODEL_CONFIG,
        "strict_years": details,
        "comparison_to_v214_joint": comparisons,
        "all_year_gains_positive": all_positive,
        "source_month_gate_passed": source_month_pass,
        "incremental_positive_years": int(incremental_years),
        "eligible_for_multiseed_confirmation": eligible,
        "eligible_for_packaging": False,
        "fitted": fitted,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.component_root,
        args.baseline_checkpoint_dir, args.joint_checkpoint_dir, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
