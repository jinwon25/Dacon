"""Leakage-safe prequential target encodings for count/platoon contexts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_GROUPS: dict[str, list[str]] = {
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
    "pitcher_batter_hand": ["pitcher_id", "batter_hand"],
    "pitcher_game_type": ["pitcher_id", "game_type"],
    "batter_count": ["batter_id", "balls_before", "strikes_before"],
    "batter_pitcher_hand": ["batter_id", "pitcher_hand"],
    "team_matchup_count": [
        "pitcher_team_id",
        "batter_team_id",
        "balls_before",
        "strikes_before",
    ],
    "count_platoon": [
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
    ],
    "pressure_platoon": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand", "base_state"],
}


@dataclass
class PrequentialTargetEncoder:
    groups: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_GROUPS))
    alpha: float = 200.0
    global_rate: float | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, target: np.ndarray | pd.Series) -> "PrequentialTargetEncoder":
        y = np.asarray(target, dtype=np.float64)
        if len(y) != len(frame):
            raise ValueError("target length differs from frame")
        self.global_rate = float(y.mean())
        self.tables = {}
        for name, columns in self.groups.items():
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(f"target encoding columns missing for {name}: {sorted(missing)}")
            key = frame[columns].copy()
            key["__n"] = 1
            key["__s"] = y
            table = key.groupby(columns, observed=True, sort=False)[["__n", "__s"]].sum().reset_index()
            table[f"te_{name}_rate"] = (
                table["__s"] + self.alpha * self.global_rate
            ) / (table["__n"] + self.alpha)
            self.tables[name] = table
        return self

    def _prequential(self, frame: pd.DataFrame, target: np.ndarray | pd.Series) -> pd.DataFrame:
        y = np.asarray(target, dtype=np.float64)
        if self.global_rate is None:
            raise RuntimeError("fit must be called first")
        out = pd.DataFrame(index=frame.index)
        for name, columns in self.groups.items():
            groups = frame.groupby(columns, observed=True, sort=False)
            prior_n = groups.cumcount().to_numpy(dtype=np.float64)
            work = frame[columns].copy()
            work["__target"] = y
            prior_s = (
                work.groupby(columns, observed=True, sort=False)["__target"]
                .cumsum()
                .to_numpy(dtype=np.float64)
                - y
            )
            out[f"te_{name}_rate"] = (
                (prior_s + self.alpha * self.global_rate)
                / (prior_n + self.alpha)
            ).astype("float32")
            out[f"te_{name}_log_n"] = np.log1p(prior_n).astype("float32")
        return out.reset_index(drop=True)

    def transform(
        self,
        frame: pd.DataFrame,
        target: np.ndarray | pd.Series | None = None,
        *,
        prequential: bool = False,
    ) -> pd.DataFrame:
        if self.global_rate is None:
            raise RuntimeError("fit must be called first")
        if prequential:
            if target is None:
                raise ValueError("target is required for prequential transform")
            return self._prequential(frame, target)
        out = pd.DataFrame(index=frame.index)
        for name, columns in self.groups.items():
            table = self.tables[name]
            left = frame[columns].reset_index(drop=True)
            merged = left.merge(
                table[columns + [f"te_{name}_rate", "__n"]],
                on=columns,
                how="left",
                sort=False,
                validate="many_to_one",
            )
            out[f"te_{name}_rate"] = merged[f"te_{name}_rate"].fillna(self.global_rate).astype("float32")
            out[f"te_{name}_log_n"] = np.log1p(merged["__n"].fillna(0.0)).astype("float32")
        return out.reset_index(drop=True)
