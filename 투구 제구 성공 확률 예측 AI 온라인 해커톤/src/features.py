"""Leakage-safe, row-local feature engineering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import ID_COL, TARGET_COL

BASE_CATEGORICAL = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

DERIVED_CATEGORICAL = [
    "count_state",
    "platoon",
    "inning_half",
    "pitcher_count",
    "pitcher_batter_hand",
    "batter_count",
    "team_matchup",
    "situation_state",
]

ASOF_COLS = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

TRACKMAN_KEYS = ["season", "balls_before", "strikes_before", "outs_before"]
TRACKMAN_MEASURES = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__")


def empirical_pitcher_prior(df: pd.DataFrame, global_rate: float) -> np.ndarray:
    rate = df["asof_pitcher_success_rate"].astype("float64").to_numpy()
    return np.where(np.isfinite(rate), rate, global_rate)


def hierarchical_prior(
    df: pd.DataFrame,
    global_rate: float,
    alpha: float = 200.0,
    batter_weight: float = 0.25,
) -> np.ndarray:
    pn = df["asof_pitcher_n"].astype("float64").to_numpy()
    bn = df["asof_batter_n"].astype("float64").to_numpy() * batter_weight
    pr = df["asof_pitcher_success_rate"].astype("float64").fillna(global_rate).to_numpy()
    br = df["asof_batter_success_rate"].astype("float64").fillna(global_rate).to_numpy()
    return (pn * pr + bn * br + alpha * global_rate) / (pn + bn + alpha)


def build_trackman_context(trackman: pd.DataFrame) -> pd.DataFrame:
    """Create compact prior-season league context keyed only by current-row state.

    For snapshot season S, only Trackman rows with season < S are included.
    This avoids 2024 leakage in the 2024 holdout and yields a 2025 lookup for test.
    """
    required = {"season", "balls_before", "strikes_before", "outs_before", "pitch_type_group", *TRACKMAN_MEASURES}
    missing = required - set(trackman.columns)
    if missing:
        raise ValueError(f"Trackman columns missing: {sorted(missing)}")

    pieces = []
    min_season = int(trackman["season"].min())
    max_season = int(trackman["season"].max())
    state_keys = ["balls_before", "strikes_before", "outs_before"]
    quantiles = [0.25, 0.50, 0.75]
    for snapshot in range(min_season + 1, max_season + 2):
        history = trackman.loc[trackman["season"] < snapshot]
        grouped = history.groupby(state_keys, observed=True)
        base = grouped.size().rename("tm_context_n").reset_index()
        for measure in TRACKMAN_MEASURES:
            stats = grouped[measure].agg(["mean", "std"]).reset_index()
            stats = stats.rename(
                columns={"mean": f"tm_{measure}_mean", "std": f"tm_{measure}_std"}
            )
            q = grouped[measure].quantile(quantiles).unstack(-1).reset_index()
            q = q.rename(
                columns={
                    0.25: f"tm_{measure}_q25",
                    0.50: f"tm_{measure}_q50",
                    0.75: f"tm_{measure}_q75",
                }
            )
            base = base.merge(stats, on=state_keys, how="left", validate="one_to_one")
            base = base.merge(q, on=state_keys, how="left", validate="one_to_one")
        pitch_counts = (
            history.groupby(state_keys + ["pitch_type_group"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for group_name in ["fastball", "breaking", "offspeed", "other"]:
            counts = pitch_counts[group_name] if group_name in pitch_counts else 0
            pitch_counts[f"tm_pitch_group_{group_name}_rate"] = counts / pitch_counts[
                [col for col in pitch_counts.columns if col not in state_keys and not str(col).startswith("tm_")]
            ].sum(axis=1)
        keep_rates = state_keys + [
            f"tm_pitch_group_{name}_rate"
            for name in ["fastball", "breaking", "offspeed", "other"]
        ]
        base = base.merge(pitch_counts[keep_rates], on=state_keys, how="left", validate="one_to_one")
        base.insert(0, "season", snapshot)
        pieces.append(base)
    context = pd.concat(pieces, ignore_index=True)
    for col in context.columns:
        if col not in TRACKMAN_KEYS:
            context[col] = context[col].astype("float32")
    return context


@dataclass
class FeatureBuilder:
    feature_set: str = "engineered"
    global_rate: float | None = None
    base_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    trackman_context: pd.DataFrame | None = None

    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.base_columns:
            self.base_columns = [col for col in df.columns if col not in {ID_COL, TARGET_COL}]
        missing = [col for col in self.base_columns if col not in df.columns]
        if missing:
            raise ValueError(f"input columns missing: {missing}")
        out = df[self.base_columns].copy()
        if self.feature_set == "no_asof":
            out = out.drop(columns=[col for col in ASOF_COLS if col in out.columns])
        if self.feature_set in {"engineered", "trackman"}:
            count = _safe_string(out["balls_before"]) + "-" + _safe_string(out["strikes_before"])
            out["count_state"] = count
            out["platoon"] = _safe_string(out["pitcher_hand"]) + "-" + _safe_string(out["batter_hand"])
            out["inning_half"] = _safe_string(out["inning"]) + "-" + _safe_string(out["top_bottom"])
            out["pitcher_count"] = _safe_string(out["pitcher_id"]) + "-" + count
            out["pitcher_batter_hand"] = _safe_string(out["pitcher_id"]) + "-" + _safe_string(out["batter_hand"])
            out["batter_count"] = _safe_string(out["batter_id"]) + "-" + count
            out["team_matchup"] = _safe_string(out["pitcher_team_id"]) + "-" + _safe_string(out["batter_team_id"])
            out["situation_state"] = count + "-" + _safe_string(out["outs_before"]) + "-" + _safe_string(out["base_state"])
            out["abs_score_diff"] = out["score_diff_pitcher_team"].abs().astype("float32")
            out["late_inning"] = (out["inning"] >= 7).astype("int8")
            out["runners_scoring_position"] = (
                (out["runner_on_2b"] > 0) | (out["runner_on_3b"] > 0)
            ).astype("int8")
            out["pitcher_log_n"] = np.log1p(out["asof_pitcher_n"]).astype("float32")
            out["batter_log_n"] = np.log1p(out["asof_batter_n"]).astype("float32")
            out["pitchmix_log_n"] = np.log1p(out["asof_pitcher_pitchmix_n"]).astype("float32")
            rate = float(self.global_rate if self.global_rate is not None else 0.5)
            pn = out["asof_pitcher_n"].astype("float64")
            bn = out["asof_batter_n"].astype("float64")
            pr = out["asof_pitcher_success_rate"].astype("float64").fillna(rate)
            br = out["asof_batter_success_rate"].astype("float64").fillna(rate)
            for alpha in (50.0, 200.0):
                out[f"pitcher_success_smooth_{int(alpha)}"] = (
                    (pn * pr + alpha * rate) / (pn + alpha)
                ).astype("float32")
            out["batter_success_smooth_200"] = ((bn * br + 200.0 * rate) / (bn + 200.0)).astype("float32")
            out["hierarchical_success_prior"] = hierarchical_prior(out, rate).astype("float32")
            out["pitcher_recent_delta_1"] = (
                out["asof_pitcher_prev1_game_success_rate"] - out["pitcher_success_smooth_200"]
            ).astype("float32")
            out["pitcher_recent_delta_3"] = (
                out["asof_pitcher_prev3_game_success_rate"] - out["pitcher_success_smooth_200"]
            ).astype("float32")
            out["pitcher_recent_delta_5"] = (
                out["asof_pitcher_prev5_game_success_rate"] - out["pitcher_success_smooth_200"]
            ).astype("float32")

        if self.feature_set == "trackman":
            if self.trackman_context is None:
                raise ValueError("trackman feature_set needs a precomputed context table")
            original_order = np.arange(len(out))
            out["__row_order"] = original_order
            out = out.merge(
                self.trackman_context,
                on=TRACKMAN_KEYS,
                how="left",
                sort=False,
                validate="many_to_one",
            ).sort_values("__row_order", kind="stable")
            out = out.drop(columns="__row_order").reset_index(drop=True)
            out["tm_context_missing"] = out["tm_context_n"].isna().astype("int8")
        return out

    def fit(self, df: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "FeatureBuilder":
        if self.global_rate is None:
            if y is None:
                raise ValueError("y is required when global_rate is not preset")
            self.global_rate = float(np.asarray(y, dtype=np.float64).mean())
        raw = self._raw_features(df)
        candidates = list(BASE_CATEGORICAL)
        if self.feature_set in {"engineered", "trackman"}:
            candidates += DERIVED_CATEGORICAL
        self.categorical_columns = [col for col in candidates if col in raw.columns]
        self.category_maps = {}
        for col in self.categorical_columns:
            values = _safe_string(raw[col])
            categories = pd.unique(values)
            self.category_maps[col] = {str(value): int(index) for index, value in enumerate(categories)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_rate is None or not self.category_maps:
            raise RuntimeError("FeatureBuilder must be fit before transform")
        out = self._raw_features(df)
        for col in self.categorical_columns:
            mapping = self.category_maps[col]
            out[col] = _safe_string(out[col]).map(mapping).fillna(-1).astype("int32")
        for col in out.columns:
            if col not in self.categorical_columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        return out

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def to_spec(self) -> dict[str, Any]:
        if self.global_rate is None:
            raise RuntimeError("cannot serialize an unfitted FeatureBuilder")
        return {
            "feature_set": self.feature_set,
            "global_rate": self.global_rate,
            "base_columns": self.base_columns,
            "categorical_columns": self.categorical_columns,
            "category_maps": self.category_maps,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any], trackman_context: pd.DataFrame | None = None) -> "FeatureBuilder":
        return cls(
            feature_set=spec["feature_set"],
            global_rate=float(spec["global_rate"]),
            base_columns=list(spec["base_columns"]),
            categorical_columns=list(spec["categorical_columns"]),
            category_maps={
                col: {str(key): int(value) for key, value in mapping.items()}
                for col, mapping in spec["category_maps"].items()
            },
            trackman_context=trackman_context,
        )

    def save_spec(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_spec(), ensure_ascii=False), encoding="utf-8")
