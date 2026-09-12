"""Leakage-audited nested v2 baseline and component decomposition.

The expensive part of the incumbent (the engineered LightGBM and official RF)
is rebuilt with a strict inner-year iteration/calibration choice.  Existing
Trackman caches are retained as a *frozen replay diagnostic* only: their outer
early-stopping metadata is never silently relabelled as nested evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset
from src.archive.data import TARGET_COL, read_main
from src.metrics import brier_score
from src.archive.train import FeatureBuilder, _base_lgb_params, train_lgb_holdout, train_rf_holdout
from src.archive.validation import walk_forward_splits


VARIANT = {
    "name": "lgb_engineered_l31",
    "feature_set": "engineered",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 500,
    "lambda_l2": 2.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "max_boost_rounds": 1200,
    "early_stopping_rounds": 100,
}
OFFICIAL_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
FROZEN_OFFSET = -0.09579592585412458


def _fit_fixed_lgb(train: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, rounds: int) -> np.ndarray:
    builder = FeatureBuilder(feature_set="engineered")
    y = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder.fit(train.iloc[train_idx], y)
    x_train = builder.transform(train.iloc[train_idx])
    x_valid = builder.transform(train.iloc[valid_idx])
    categorical = [col for col in builder.categorical_columns if col in x_train.columns]
    dtrain = lgb.Dataset(x_train, label=y, categorical_feature=categorical, free_raw_data=True)
    booster = lgb.train(
        _base_lgb_params(VARIANT), dtrain, num_boost_round=max(1, int(rounds)),
        callbacks=[lgb.log_evaluation(0)],
    )
    result = np.asarray(booster.predict(x_valid, num_iteration=int(rounds)), dtype=np.float64)
    del builder, x_train, x_valid, dtrain, booster
    gc.collect()
    return np.clip(result, 1e-6, 1.0 - 1e-6)


def _select_offset(pred: np.ndarray, y: np.ndarray) -> float:
    """Select an intercept on inner validation only (deterministic grid)."""
    candidates = np.linspace(-0.25, 0.25, 1001)
    clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    scores = np.asarray([
        np.mean((1.0 / (1.0 + np.exp(-np.clip(logits + value, -40.0, 40.0))) - y) ** 2)
        for value in candidates
    ])
    return float(candidates[int(np.argmin(scores))])


def _recency_weights(season: np.ndarray, outer_year: int, half_life: float = 1.0) -> np.ndarray:
    return np.power(0.5, (outer_year - season.astype(float)) / half_life).astype(np.float64)


def _load_frozen(project: Path, year: int) -> dict[str, np.ndarray]:
    path = project / "artifacts" / "followup" / "oof" / f"wave0_incumbent_validate_{year}.npz"
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def run(project: Path, years: tuple[int, ...] = (2021, 2022, 2023, 2024)) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    started = time.perf_counter()
    train = read_main(project / "data" / "train.csv")
    folds = {fold.validation_season: fold for fold in walk_forward_splits(train, validation_seasons=years)}
    rows: list[dict] = []
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for year in years:
        fold = folds[year]
        inner_year = year - 1
        inner = folds.get(inner_year)
        if inner is None:
            # 2020 was already run in the legacy cache and is used only as the
            # pre-registered inner validation for the first primary fold.
            inner_train = np.flatnonzero(train["season"].to_numpy() < inner_year)
            inner_valid = np.flatnonzero(train["season"].to_numpy() == inner_year)
        else:
            inner_train, inner_valid = inner.train_idx, inner.valid_idx
        if len(inner_train) == 0 or len(inner_valid) == 0:
            raise ValueError(f"missing inner fold for outer year {year}")

        # Iteration selection sees only (inner_train, inner_valid).
        inner_lgb = train_lgb_holdout(
            train, inner_train, inner_valid, VARIANT,
            int(VARIANT["max_boost_rounds"]), int(VARIANT["early_stopping_rounds"]), None,
        )
        inner_rf = train_rf_holdout(train, inner_train, inner_valid, OFFICIAL_FEATURES)
        inner_blend = 0.35 * np.asarray(inner_lgb["prediction"]) + 0.65 * np.asarray(inner_rf["prediction"])
        inner_offset = _select_offset(inner_blend, train.iloc[inner_valid][TARGET_COL].to_numpy(dtype=float))
        rounds = int(inner_lgb["best_iteration"])

        # Outer target is not passed to either callback or selection routine.
        lgb_pred = _fit_fixed_lgb(train, fold.train_idx, fold.valid_idx, rounds)
        rf = train_rf_holdout(train, fold.train_idx, fold.valid_idx, OFFICIAL_FEATURES)
        rf_pred = np.asarray(rf["prediction"], dtype=np.float64)
        lgb_cal = apply_logit_offset(lgb_pred, inner_offset)
        rf_cal = apply_logit_offset(rf_pred, inner_offset)
        base = np.clip(0.35 * lgb_cal + 0.65 * rf_cal, 1e-4, 1.0 - 1e-4)

        recency = train_rf_holdout(
            train, fold.train_idx, fold.valid_idx, OFFICIAL_FEATURES,
            sample_weight=_recency_weights(train.iloc[fold.train_idx]["season"].to_numpy(), year),
        )
        recency_pred = np.asarray(recency["prediction"], dtype=np.float64)
        recency_arm = np.clip(0.35 * lgb_cal + 0.65 * apply_logit_offset(recency_pred, inner_offset), 1e-4, 1.0 - 1e-4)
        target = train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=float)

        predictions[year] = {
            "target": target,
            "A_original": base,
            "B_recency": recency_arm,
            "lgb": lgb_cal,
            "rf": rf_cal,
            "valid_idx": fold.valid_idx,
        }
        for arm, pred in (("A_original", base), ("B_R_recency", recency_arm)):
            rows.append({
                "baseline": "v2_nested",
                "arm": arm,
                "outer_validation_season": year,
                "n_rows": len(target),
                "brier": brier_score(target, pred),
                "delta_vs_A": brier_score(target, pred) - brier_score(target, base),
                "inner_validation_season": inner_year,
                "selected_lgb_iterations": rounds,
                "selected_offset": inner_offset,
                "outer_target_used_for_selection": False,
                "prediction_source": "strict_inner_selection_then_outer_fixed_fit",
            })

        # C/D are a diagnostic replay from the existing package cache. They
        # are intentionally flagged non-primary because their historical
        # Trackman callback used the outer target year for early stopping.
        frozen = _load_frozen(project, year)
        tm_path = project / "artifacts" / "followup" / "models" / f"lgb_trackman_pitcher_v1_validate_{year}.npz"
        if tm_path.exists():
            with np.load(tm_path, allow_pickle=False) as tm:
                tm_pred = np.asarray(tm["prediction"], dtype=float)
            c = np.clip(0.95 * base + 0.05 * tm_pred, 1e-4, 1.0 - 1e-4)
            for arm, pred in (("C_trackman_5pct", c), ("D_full_v2_replay", c)):
                rows.append({
                    "baseline": "v2_frozen_replay_diagnostic",
                    "arm": arm,
                    "outer_validation_season": year,
                    "n_rows": len(target),
                    "brier": brier_score(target, pred),
                    "delta_vs_A": brier_score(target, pred) - brier_score(target, base),
                    "inner_validation_season": inner_year,
                    "selected_lgb_iterations": rounds,
                    "selected_offset": inner_offset,
                    "outer_target_used_for_selection": True,
                    "prediction_source": "legacy_trackman_outer_early_stopping; diagnostic_only",
                })
        del inner_lgb, inner_rf, rf, recency
        gc.collect()

    result = pd.DataFrame(rows)
    report_path = project / "research" / "reports" / "champion_v2_nested_results.csv"
    result.to_csv(report_path, index=False)
    np.savez_compressed(project / "artifacts" / "followup" / "v2_nested_predictions.npz", **{
        f"{year}_{key}": value for year, values in predictions.items() for key, value in values.items()
    })
    (project / "artifacts" / "followup" / "v2_nested_provenance.json").write_text(
        json.dumps({
            "rule": "inner train < outer-1; inner validation = outer-1; fixed fit season < outer",
            "outer_target_used_for_selection": False,
            "runtime_seconds": time.perf_counter() - started,
            "note": "A/B are strict nested. C/D reuse legacy Trackman predictions only for decomposition diagnostics and cannot promote.",
        }, indent=2), encoding="utf-8",
    )
    print(result.to_string(index=False))
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.project_dir.resolve())


if __name__ == "__main__":
    main()
