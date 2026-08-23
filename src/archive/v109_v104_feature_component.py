"""Standalone row-local feature component for the v104 public probe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-6
CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "platoon",
    "team_matchup",
)
CONDITIONAL_COLUMNS = (
    "prior_pitcher_rate",
    "prior_pitcher_n",
    "prior_hand_rate",
    "prior_hand_dev",
    "prior_hand_n",
    "prior_hand_reliability",
    "prior_count_rate",
    "prior_count_dev",
    "prior_count_n",
    "prior_count_reliability",
)
EXCLUDED = {
    "row_id",
    "control_success",
    "season",
    "game_month",
    "pitcher_id",
    "batter_id",
}


def count_state(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["balls_before"].astype("Int64").astype("string")
        + "-"
        + frame["strikes_before"].astype("Int64").astype("string")
    )


def _aggregate(history: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        history.groupby(keys, observed=True, sort=False)["control_success"]
        .agg(success="sum", n="size")
        .reset_index()
    )


def build_bank(history: pd.DataFrame, strengths: dict[str, float]) -> dict[str, Any]:
    if history.empty:
        return {"global": 0.5, "pitcher": None, "hand": None, "count": None}
    work = history[["pitcher_id", "batter_hand", "control_success"]].copy()
    work["count_state"] = count_state(history).to_numpy()
    global_rate = float(work["control_success"].mean())
    pitcher = _aggregate(work, ["pitcher_id"])
    alpha = float(strengths["pitcher"])
    pitcher["rate"] = (pitcher["success"] + alpha * global_rate) / (
        pitcher["n"] + alpha
    )

    def child(keys: list[str], strength: float) -> pd.DataFrame:
        table = _aggregate(work, keys)
        table = table.merge(
            pitcher[["pitcher_id", "rate"]].rename(columns={"rate": "parent"}),
            on="pitcher_id",
            how="left",
            validate="many_to_one",
        )
        table["parent"] = table["parent"].fillna(global_rate)
        table["rate"] = (
            table["success"] + float(strength) * table["parent"]
        ) / (table["n"] + float(strength))
        table["dev"] = table["rate"] - table["parent"]
        table["reliability"] = table["n"] / (table["n"] + float(strength))
        return table

    return {
        "global": global_rate,
        "pitcher": pitcher,
        "hand": child(
            ["pitcher_id", "batter_hand"],
            float(strengths["pitcher_batter_hand"]),
        ),
        "count": child(
            ["pitcher_id", "count_state"],
            float(strengths["pitcher_count"]),
        ),
    }


def _lookup(
    rows: pd.DataFrame,
    table: pd.DataFrame | None,
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame(
            {name: np.zeros(len(rows), dtype=np.float64) for name in values}
        )
    merged = rows[keys].reset_index(drop=True).merge(
        table[keys + values],
        on=keys,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    return merged[values]


def feature_frame(rows: pd.DataFrame, bank: dict[str, Any]) -> pd.DataFrame:
    out = rows.drop(
        columns=[column for column in EXCLUDED if column in rows], errors="ignore"
    ).copy()
    out["count_state"] = count_state(rows).astype("string")
    out["platoon"] = (
        rows["pitcher_hand"].astype("string")
        + "-"
        + rows["batter_hand"].astype("string")
    )
    out["team_matchup"] = (
        rows["pitcher_team_id"].astype("string")
        + "-"
        + rows["batter_team_id"].astype("string")
    )
    out["same_hand"] = rows["pitcher_hand"].eq(rows["batter_hand"]).astype(np.int8)

    global_rate = float(bank["global"])
    p_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0).to_numpy()
    p_rate = pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce").to_numpy()
    b_n = pd.to_numeric(rows["asof_batter_n"], errors="coerce").fillna(0.0).to_numpy()
    b_rate = pd.to_numeric(rows["asof_batter_success_rate"], errors="coerce").to_numpy()
    p_rate = np.where(np.isfinite(p_rate), p_rate, global_rate)
    b_rate = np.where(np.isfinite(b_rate), b_rate, global_rate)
    out["shrunk_pitcher_rate"] = (p_n * p_rate + 300.0 * global_rate) / (p_n + 300.0)
    out["shrunk_batter_rate"] = (b_n * b_rate + 300.0 * global_rate) / (b_n + 300.0)
    for window in (1, 3, 5):
        recent = pd.to_numeric(
            rows[f"asof_pitcher_prev{window}_game_success_rate"], errors="coerce"
        ).to_numpy()
        out[f"form_diff_{window}"] = np.where(np.isfinite(recent), recent - p_rate, 0.0)
    out["ball_minus_strike_rate"] = (
        pd.to_numeric(rows["asof_pitcher_ball_rate"], errors="coerce")
        - pd.to_numeric(rows["asof_pitcher_strike_rate"], errors="coerce")
    )
    mix = rows[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    out["pitchmix_entropy"] = -np.sum(
        np.where(mix > 0.0, mix * np.log(np.clip(mix, EPS, 1.0)), 0.0), axis=1
    )

    keys = rows[["pitcher_id", "batter_hand"]].copy()
    keys["count_state"] = count_state(rows).to_numpy()
    pitcher = _lookup(keys, bank["pitcher"], ["pitcher_id"], ["rate", "n"])
    hand = _lookup(
        keys,
        bank["hand"],
        ["pitcher_id", "batter_hand"],
        ["rate", "dev", "n", "reliability"],
    )
    count = _lookup(
        keys,
        bank["count"],
        ["pitcher_id", "count_state"],
        ["rate", "dev", "n", "reliability"],
    )
    out["prior_pitcher_rate"] = pitcher["rate"].fillna(global_rate).to_numpy()
    out["prior_pitcher_n"] = pitcher["n"].fillna(0.0).to_numpy()
    for prefix, block in (("prior_hand", hand), ("prior_count", count)):
        out[f"{prefix}_rate"] = block["rate"].fillna(global_rate).to_numpy()
        out[f"{prefix}_dev"] = block["dev"].fillna(0.0).to_numpy()
        out[f"{prefix}_n"] = block["n"].fillna(0.0).to_numpy()
        out[f"{prefix}_reliability"] = block["reliability"].fillna(0.0).to_numpy()
    for column in CATEGORICAL:
        out[column] = out[column].astype("string").fillna("__MISSING__")
    if "game_month" in out or "pitcher_id" in out or "batter_id" in out:
        raise AssertionError("prohibited feature survived construction")
    return out


def align_categories(
    fit: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    fit = fit.copy()
    audit = audit.copy()
    categories: dict[str, list[str]] = {}
    for column in CATEGORICAL:
        values = pd.Index(fit[column].astype(str).unique()).tolist()
        categories[column] = [str(value) for value in values]
        fit[column] = pd.Categorical(fit[column].astype(str), categories=values)
        audit[column] = pd.Categorical(audit[column].astype(str), categories=values)
    columns = sorted(fit.columns)
    return fit[columns], audit[columns], categories


def apply_model_spec(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    for column, values in spec["categories"].items():
        output[column] = pd.Categorical(
            output[column].astype("string").fillna("__MISSING__").astype(str),
            categories=[str(value) for value in values],
        )
    columns = [str(value) for value in spec["feature_columns"]]
    missing = sorted(set(columns) - set(output.columns))
    if missing:
        raise ValueError(f"inference features missing: {missing}")
    return output[columns]
