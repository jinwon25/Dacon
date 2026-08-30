"""Frozen fallback-XGB transform with a prior-history end anchor.

The ASOF counters in the competition data are career-cumulative.  A future
season delta must therefore subtract the latest observable pre-pitch state in
official training history, not the first state of the latest training season.
Every transformation below is row-local at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


TARGET = "control_success"
CAT = [
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
SPECS = [
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "p_succ"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "p_rev"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", "p_mid"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", "p_ball"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", "p_stk"),
    ("batter_id", "asof_batter_n", "asof_batter_success_rate", "b_succ"),
    ("batter_id", "asof_batter_n", "asof_batter_middle_rate", "b_mid"),
]
SITS = [
    "3ball",
    "2strk",
    "ahead",
    "behind",
    "risp",
    "on1b",
    "vsL",
    "vsR",
    "late",
    "hiLI",
    "loLI",
    "blowout",
]
TM = [
    "rel_speed",
    "rel_speed_sd",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
]


def _anchor_arrays(
    ids: np.ndarray, pairs: dict[int, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    values = [pairs.get(int(value), (0.0, 0.0)) for value in ids]
    return (
        np.asarray([value[0] for value in values], dtype=np.float64),
        np.asarray([value[1] for value in values], dtype=np.float64),
    )


def build(frame: pd.DataFrame, asset: Path) -> pd.DataFrame:
    lookup = joblib.load(asset / "fallback_lookups.joblib")
    if int(lookup.get("features_version", 0)) < 2:
        raise ValueError("exact-anchor fallback lookup is required")
    if lookup.get("anchor_semantics") != "latest_prior_pre_pitch_state":
        raise ValueError("unexpected fallback anchor semantics")

    output = frame.drop(
        columns=["row_id", TARGET, "asof_pitcher_pitchmix_n"], errors="ignore"
    ).copy()
    for column in CAT:
        output[column] = (
            frame[column]
            .fillna("__NA__")
            .astype(str)
            .map(lookup["cat"][column])
            .fillna(-1)
            .astype(np.float32)
        )
    output["hand_mix"] = output["pitcher_hand"] * 2 + output["batter_hand"]

    exact_state: dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    for id_column, n_column, rate_column, prefix in SPECS:
        ids = frame[id_column].astype(int).to_numpy()
        anchor_n, anchor_sum = _anchor_arrays(ids, lookup["anchors"][prefix])
        n = pd.to_numeric(frame[n_column], errors="coerce").to_numpy(np.float64)
        rate = pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(
            np.float64
        )
        prior = float(lookup["priors"][prefix])
        cumulative_sum = np.nan_to_num(n * rate, nan=0.0)
        delta_n = np.maximum(np.nan_to_num(n, nan=0.0) - anchor_n, 0.0)
        delta_sum = np.maximum(cumulative_sum - anchor_sum, 0.0)
        season_rate = (delta_sum + 150.0 * prior) / (delta_n + 150.0)
        output[f"{prefix}_ssn"] = season_rate
        output[f"{prefix}_ssn_vs_car"] = season_rate - np.nan_to_num(
            rate, nan=prior
        )
        if prefix in ("p_succ", "b_succ"):
            output[f"{prefix}_ssn_n"] = delta_n
        exact_state[prefix] = (delta_n, delta_sum, prior, rate)

    output["p_prev5_vs_car"] = (
        frame["asof_pitcher_prev5_game_success_rate"]
        - frame["asof_pitcher_success_rate"]
    )
    output["p_prev1_vs_prev5"] = (
        frame["asof_pitcher_prev1_game_success_rate"]
        - frame["asof_pitcher_prev5_game_success_rate"]
    )

    # The original release used career totals here.  These multiscale features
    # must use the same exact end anchor as the season-rate features.
    for prefix in ("p_succ", "b_succ"):
        delta_n, delta_sum, prior, _rate = exact_state[prefix]
        for shrink in (25, 75, 400, 1000):
            output[f"{prefix}_k{shrink}"] = (
                delta_sum + shrink * prior
            ) / (delta_n + shrink)

    pitcher_ids = frame["pitcher_id"].astype(int).to_numpy()
    overall = np.asarray(
        [lookup["overall"].get(int(value), np.nan) for value in pitcher_ids]
    )
    for name in SITS:
        rate = np.asarray(
            [
                lookup["situations"][name].get(int(value), np.nan)
                for value in pitcher_ids
            ]
        )
        output[f"p_sit_{name}"] = rate
    output["p_sit_overall"] = overall
    for name in SITS:
        output[f"p_sit_{name}_d"] = output[f"p_sit_{name}"] - overall

    masks = [
        frame["balls_before"].eq(3),
        frame["strikes_before"].eq(2),
        frame["strikes_before"].gt(frame["balls_before"]),
        frame["balls_before"].gt(frame["strikes_before"]),
        frame["runner_on_2b"].eq(1) | frame["runner_on_3b"].eq(1),
        frame["runner_on_1b"].eq(1),
        frame["batter_hand"].eq(1),
        frame["batter_hand"].eq(2),
        frame["inning"].ge(7),
        frame["li"].gt(1.5),
        frame["li"].lt(0.5),
        frame["score_diff_pitcher_team"].abs().ge(5),
    ]
    matched = np.full(len(frame), np.nan)
    for name, mask in zip(SITS, masks):
        delta = output[f"p_sit_{name}_d"].to_numpy()
        selected = mask.to_numpy() & np.isnan(matched)
        matched[selected] = delta[selected]
    output["p_sit_matched"] = matched

    pair_key = (
        frame["pitcher_id"].astype("int64") * 100000
        + frame["batter_id"].astype("int64")
    ).to_numpy()
    pair_values = [lookup["pb"].get(int(value), (0.0, 0.0)) for value in pair_key]
    pair_sum = np.asarray([value[0] for value in pair_values])
    pair_count = np.asarray([value[1] for value in pair_values])
    output["pb_n"] = pair_count
    output["pb_rate"] = (
        pair_sum + 30.0 * lookup["target_mean"]
    ) / (pair_count + 30.0)
    output["pb_logn"] = np.log1p(pair_count)

    ppa = np.asarray(
        [
            lookup["ppa"].get(int(value), lookup["ppa_default"])
            for value in pitcher_ids
        ]
    )
    output["p_ppa"] = ppa
    output["p_est_apps"] = output["p_succ_ssn_n"] / np.clip(ppa, 5.0, None)
    output["p_inning_x_role"] = (
        frame["inning"].to_numpy(np.float64) * np.log1p(ppa)
    )
    output["p_ssn_per_month"] = output["p_succ_ssn_n"] / np.clip(
        frame["game_month"].to_numpy(np.float64), 3.0, None
    )
    for column in TM:
        output[f"tm_{column}"] = np.asarray(
            [lookup["tm"][column].get(int(value), np.nan) for value in pitcher_ids]
        )

    columns = json.loads(
        (asset / "feature_columns.json").read_text(encoding="utf-8")
    )
    return (
        output.apply(pd.to_numeric, errors="coerce")
        .astype(np.float32)
        .reindex(columns=columns)
    )


def predict(frame: pd.DataFrame, asset: Path) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(asset / "fallback_xgb.json")
    return model.predict_proba(build(frame, asset))[:, 1]
