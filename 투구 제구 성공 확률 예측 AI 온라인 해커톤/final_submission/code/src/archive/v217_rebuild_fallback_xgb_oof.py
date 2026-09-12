"""Rebuild the Public-1175 fallback XGB strict-forward OOF contract.

The team 114-feature contract is reconstructed from frozen runtime assets.
This module keeps the strict-forward OOF feature contract explicit from the
team feature definitions and the frozen runtime assets.  Labels are
used only from seasons strictly before each audit season.
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
import xgboost as xgb
from common_features.season import multi_scale_success


PROTOCOL = "V217_FALLBACK_XGB_STRICT_OOF_REBUILD_V1"
TARGET = "control_success"
CAT = (
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
)
SPECS = (
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "p_succ"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "p_rev"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", "p_mid"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", "p_ball"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", "p_stk"),
    ("batter_id", "asof_batter_n", "asof_batter_success_rate", "b_succ"),
    ("batter_id", "asof_batter_n", "asof_batter_middle_rate", "b_mid"),
)
SITUATIONS = (
    "3ball", "2strk", "ahead", "behind", "risp", "on1b",
    "vsL", "vsR", "late", "hiLI", "loLI", "blowout",
)
TM_COLUMNS = (
    "rel_speed", "rel_speed_sd", "spin_rate", "induced_vert_break",
    "horz_break", "extension", "rel_height", "rel_side",
)
YEARS = (2022, 2023, 2024)
PARAMS = {
    "n_estimators": 1800,
    "learning_rate": 0.006,
    "max_depth": 10,
    "min_child_weight": 6000,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 50.0,
    "reg_alpha": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 16,
    "eval_metric": "logloss",
    "verbosity": 0,
}


def situation_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "3ball": frame["balls_before"].eq(3).to_numpy(),
        "2strk": frame["strikes_before"].eq(2).to_numpy(),
        "ahead": frame["strikes_before"].gt(frame["balls_before"]).to_numpy(),
        "behind": frame["balls_before"].gt(frame["strikes_before"]).to_numpy(),
        "risp": (
            frame["runner_on_2b"].eq(1) | frame["runner_on_3b"].eq(1)
        ).to_numpy(),
        "on1b": frame["runner_on_1b"].eq(1).to_numpy(),
        "vsL": frame["batter_hand"].eq(1).to_numpy(),
        "vsR": frame["batter_hand"].eq(2).to_numpy(),
        "late": frame["inning"].ge(7).to_numpy(),
        "hiLI": frame["li"].gt(1.5).to_numpy(),
        "loLI": frame["li"].lt(0.5).to_numpy(),
        "blowout": frame["score_diff_pitcher_team"].abs().ge(5).to_numpy(),
    }


def _anchor_table(
    frame: pd.DataFrame, id_column: str, n_column: str, rate_column: str
) -> pd.DataFrame:
    values = frame[[id_column, "season", n_column, rate_column]].copy()
    values["succ"] = values[n_column] * values[rate_column].fillna(0.0)
    index = values.groupby([id_column, "season"], sort=False)[n_column].idxmin()
    return values.loc[index, [id_column, "season", n_column, "succ"]]


def _season_features(frame: pd.DataFrame, output: pd.DataFrame) -> None:
    for id_column, n_column, rate_column, prefix in SPECS:
        anchor = _anchor_table(frame, id_column, n_column, rate_column)
        lookup = frame[[id_column, "season"]].merge(
            anchor, on=[id_column, "season"], how="left", sort=False
        )
        n = pd.to_numeric(frame[n_column], errors="coerce").to_numpy(np.float64)
        rate = pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(np.float64)
        anchor_n = lookup[n_column].fillna(0.0).to_numpy(np.float64)
        anchor_success = lookup["succ"].fillna(0.0).to_numpy(np.float64)
        delta_n = np.maximum(n - anchor_n, 0.0)
        delta_success = np.maximum(np.nan_to_num(n * rate) - anchor_success, 0.0)
        prior = float(frame[rate_column].mean())
        output[f"{prefix}_ssn"] = (delta_success + 150.0 * prior) / (
            delta_n + 150.0
        )
        output[f"{prefix}_ssn_vs_car"] = output[f"{prefix}_ssn"] - np.nan_to_num(
            rate, nan=prior
        )
        if prefix in ("p_succ", "b_succ"):
            output[f"{prefix}_ssn_n"] = delta_n

    for name, values in multi_scale_success(frame).items():
        output[name] = values

    output["p_prev5_vs_car"] = (
        frame["asof_pitcher_prev5_game_success_rate"]
        - frame["asof_pitcher_success_rate"]
    )
    output["p_prev1_vs_prev5"] = (
        frame["asof_pitcher_prev1_game_success_rate"]
        - frame["asof_pitcher_prev5_game_success_rate"]
    )


def _appearance_table(frame: pd.DataFrame) -> pd.DataFrame:
    game_id = frame["inning"].diff().fillna(0).lt(0).cumsum()
    return (
        frame.assign(_game_id=game_id)
        .groupby(["_game_id", "pitcher_id", "season"], sort=False)
        .size()
        .groupby(["pitcher_id", "season"], sort=False)
        .median()
        .rename("ppa")
        .reset_index()
    )


def _trackman_season_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame
) -> pd.DataFrame:
    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(0.90), ["pitcher_id", "pitcher_trackman_id"]
    ]
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    means = (
        joined.groupby(["pitcher_id", "season"], sort=False)[
            [column for column in TM_COLUMNS if column != "rel_speed_sd"]
        ]
        .mean()
    )
    means["rel_speed_sd"] = joined.groupby(
        ["pitcher_id", "season"], sort=False
    )["rel_speed"].std()
    return means.reset_index()


def _forward_context_features(
    frame: pd.DataFrame,
    output: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
) -> None:
    season = frame["season"].to_numpy(np.int16)
    target = frame[TARGET].to_numpy(np.float64)
    pitcher = frame["pitcher_id"].to_numpy(np.int64)
    batter = frame["batter_id"].to_numpy(np.int64)
    masks = situation_masks(frame)
    appearances = _appearance_table(frame)
    tm_profiles = _trackman_season_profiles(trackman, pitcher_map)

    context_columns = ["p_sit_overall"]
    context_columns += [f"p_sit_{name}" for name in SITUATIONS]
    context_columns += [f"p_sit_{name}_d" for name in SITUATIONS]
    context_columns += [
        "p_sit_matched", "pb_n", "pb_rate", "pb_logn", "p_ppa",
        "p_est_apps", "p_inning_x_role", "p_ssn_per_month",
    ]
    context_columns += [f"tm_{column}" for column in TM_COLUMNS]
    for column in context_columns:
        output[column] = np.nan

    pair_key = pitcher * 100000 + batter
    for year in sorted(np.unique(season)):
        audit = season == year
        history = season < year
        if not history.any():
            continue
        history_frame = frame.loc[history]
        history_mean = float(target[history].mean())
        total = history_frame.groupby("pitcher_id", sort=False)[TARGET].agg(
            ["sum", "count"]
        )
        raw_overall = total["sum"] / total["count"]
        smooth_overall = (
            total["sum"] + 300.0 * history_mean
        ) / (total["count"] + 300.0)
        audit_pitcher = pd.Series(pitcher[audit])
        # Materialize rates as float32 before subtracting. This precision order
        # agrees with every stored split threshold in the submitted booster.
        overall = audit_pitcher.map(smooth_overall).to_numpy(np.float32)
        output.loc[audit, "p_sit_overall"] = overall
        matched = np.full(int(audit.sum()), np.nan, dtype=np.float64)
        for name in SITUATIONS:
            selected = history & masks[name]
            table = frame.loc[selected].groupby("pitcher_id", sort=False)[TARGET].agg(
                ["sum", "count"]
            ).reindex(total.index).fillna(0.0)
            rate = (table["sum"] + 300.0 * raw_overall) / (
                table["count"] + 300.0
            )
            value = audit_pitcher.map(rate).to_numpy(np.float32)
            delta = value - overall
            output.loc[audit, f"p_sit_{name}"] = value
            output.loc[audit, f"p_sit_{name}_d"] = delta
            take = masks[name][audit] & np.isnan(matched)
            matched[take] = delta[take]
        output.loc[audit, "p_sit_matched"] = matched

        pair = pd.DataFrame({"key": pair_key[history], TARGET: target[history]}).groupby(
            "key", sort=False
        )[TARGET].agg(["sum", "count"])
        audit_pair = pd.Series(pair_key[audit])
        pair_count = audit_pair.map(pair["count"]).fillna(0.0).to_numpy(np.float64)
        pair_success = audit_pair.map(pair["sum"]).fillna(0.0).to_numpy(np.float64)
        output.loc[audit, "pb_n"] = pair_count
        output.loc[audit, "pb_rate"] = (
            pair_success + 30.0 * history_mean
        ) / (pair_count + 30.0)
        output.loc[audit, "pb_logn"] = np.log1p(pair_count)

        prior_apps = appearances.loc[appearances["season"].lt(year)]
        ppa = prior_apps.groupby("pitcher_id", sort=False)["ppa"].median()
        ppa_default = float(prior_apps["ppa"].median())
        ppa_value = audit_pitcher.map(ppa).fillna(ppa_default).to_numpy(np.float64)
        output.loc[audit, "p_ppa"] = ppa_value
        season_n = output.loc[audit, "p_succ_ssn_n"].to_numpy(np.float64)
        output.loc[audit, "p_est_apps"] = season_n / np.clip(ppa_value, 5.0, None)
        output.loc[audit, "p_inning_x_role"] = (
            frame.loc[audit, "inning"].to_numpy(np.float32)
            * np.log1p(ppa_value.astype(np.float32))
        )
        output.loc[audit, "p_ssn_per_month"] = season_n / np.clip(
            frame.loc[audit, "game_month"].to_numpy(np.float64), 3.0, None
        )

        prior_tm = tm_profiles.loc[tm_profiles["season"].lt(year)]
        tm_lookup = prior_tm.groupby("pitcher_id", sort=False)[list(TM_COLUMNS)].mean()
        for column in TM_COLUMNS:
            output.loc[audit, f"tm_{column}"] = audit_pitcher.map(
                tm_lookup[column]
            ).to_numpy(np.float64)


def build_features(
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    pitcher_map: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = train.drop(
        columns=["row_id", TARGET, "asof_pitcher_pitchmix_n"], errors="ignore"
    ).copy()
    for column in CAT:
        mapping = {
            str(value): index
            for index, value in enumerate(
                train[column].fillna("__NA__").astype(str).unique()
            )
        }
        output[column] = (
            train[column].fillna("__NA__").astype(str).map(mapping).fillna(-1)
        )
    output["hand_mix"] = output["pitcher_hand"] * 2 + output["batter_hand"]
    _season_features(train, output)
    _forward_context_features(train, output, trackman, pitcher_map)
    return (
        output.apply(pd.to_numeric, errors="coerce")
        .astype(np.float32)
        .reindex(columns=feature_columns)
    )


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean((target - prediction) ** 2) / (rate * (1.0 - rate)))
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "strictly_prior_season_label_lookups": True,
        "strictly_prior_season_trackman_profiles": True,
        "row_local_current_season_features": True,
        "fixed_public1175_recipe": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    pitcher_map_csv: Path,
    feature_columns_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    trackman = pd.read_csv(trackman_csv, encoding="utf-8-sig", low_memory=False)
    pitcher_map = pd.read_csv(pitcher_map_csv)
    feature_columns = json.loads(feature_columns_json.read_text(encoding="utf-8"))
    cache = output_dir / "features_114.parquet"
    if cache.exists():
        features = pd.read_parquet(cache)
        if list(features.columns) != feature_columns or len(features) != len(train):
            raise ValueError("cached feature contract mismatch")
        feature_reused = True
    else:
        started = time.time()
        features = build_features(train, trackman, pitcher_map, feature_columns)
        features.to_parquet(cache, index=False)
        print(
            f"[v217] built features rows={len(features):,} cols={features.shape[1]} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
        feature_reused = False

    target = train[TARGET].to_numpy(np.float32)
    season = train["season"].to_numpy(np.int16)
    is_futures = train["game_type"].astype(str).eq("F").to_numpy()
    folds: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        checkpoint = output_dir / f"fallback_xgb_oof_{year}.npy"
        validation = (season == year) & ~is_futures
        if checkpoint.exists():
            prediction = np.load(checkpoint, allow_pickle=False).astype(np.float64)
            if len(prediction) != int(validation.sum()):
                raise ValueError(f"checkpoint length mismatch: {year}")
            reused = True
            elapsed = 0.0
        else:
            fit = (season < year) & ~(is_futures & (season <= 2022))
            weight = (
                0.5 ** ((year - 1 - season[fit]) / 2.0)
            ).astype(np.float32)
            model = xgb.XGBClassifier(**PARAMS, random_state=2028)
            started = time.time()
            model.fit(features.loc[fit], target[fit], sample_weight=weight)
            prediction = model.predict_proba(features.loc[validation])[:, 1]
            elapsed = time.time() - started
            np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
            reused = False
            print(
                f"[v217] fold={year} fit={int(fit.sum()):,} "
                f"audit={int(validation.sum()):,} bss={bss(target[validation], prediction):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        folds[str(year)] = {
            "fit_rows": int(((season < year) & ~(is_futures & (season <= 2022))).sum()),
            "audit_rows": int(validation.sum()),
            "bss": bss(target[validation], prediction),
            "elapsed_seconds": float(elapsed),
            "reused": reused,
            "checkpoint": str(checkpoint),
        }

    summary = {
        "protocol": PROTOCOL,
        "status": "oof_rebuilt",
        "feature_count": len(feature_columns),
        "feature_cache": str(cache),
        "feature_cache_reused": feature_reused,
        "model_params": PARAMS,
        "folds": folds,
        "eligible_for_public1175_incremental_audit": True,
        "eligible_for_packaging": False,
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
    parser.add_argument("--pitcher-map-csv", type=Path, required=True)
    parser.add_argument("--feature-columns-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.trackman_csv,
        args.pitcher_map_csv,
        args.feature_columns_json,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
