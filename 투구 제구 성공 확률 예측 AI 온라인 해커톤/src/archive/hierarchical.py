"""Train-only empirical-Bayes entity/context backoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL


def add_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only deterministic, row-local context keys."""
    out = frame.copy()
    out["count_state"] = (
        out["balls_before"].astype("string")
        + "-"
        + out["strikes_before"].astype("string")
    )
    out["inning_bucket"] = pd.cut(
        pd.to_numeric(out["inning"], errors="coerce"),
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype("string")
    out["leverage_bucket"] = pd.cut(
        pd.to_numeric(out["li"], errors="coerce"),
        bins=[-np.inf, 0.75, 1.5, 3.0, np.inf],
        labels=["low", "medium", "high", "very_high"],
    ).astype("string")
    return out


@dataclass(frozen=True)
class BackoffSpec:
    name: str
    leaf_keys: tuple[str, ...]
    parent_keys: tuple[str, ...]
    weight: float


DEFAULT_SPECS = (
    BackoffSpec("pitcher_batter_hand", ("pitcher_id", "batter_hand"), ("pitcher_id",), 0.25),
    BackoffSpec("batter_pitcher_hand", ("batter_id", "pitcher_hand"), ("batter_id",), 0.10),
    BackoffSpec("pitcher_count", ("pitcher_id", "count_state"), ("pitcher_id",), 0.20),
    BackoffSpec("pitcher_game_type", ("pitcher_id", "game_type"), ("pitcher_id",), 0.10),
    BackoffSpec(
        "pitcher_team_game_type",
        ("pitcher_team_id", "game_type"),
        ("game_type",),
        0.10,
    ),
    BackoffSpec(
        "count_base_state",
        ("count_state", "base_state"),
        ("count_state",),
        0.15,
    ),
    BackoffSpec(
        "inning_runner",
        ("inning_bucket", "base_state"),
        ("base_state",),
        0.05,
    ),
    BackoffSpec(
        "leverage_runner",
        ("leverage_bucket", "base_state"),
        ("base_state",),
        0.05,
    ),
)


def _statistics(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    return (
        frame.groupby(list(keys), observed=True, dropna=False)[TARGET_COL]
        .agg(["sum", "count"])
        .reset_index()
    )


class HierarchicalBackoff:
    def __init__(
        self,
        alpha_leaf: float = 100.0,
        alpha_parent: float = 200.0,
        specs: tuple[BackoffSpec, ...] = DEFAULT_SPECS,
    ):
        self.alpha_leaf = float(alpha_leaf)
        self.alpha_parent = float(alpha_parent)
        self.specs = specs
        self.global_rate: float | None = None
        self.tables: dict[str, dict[str, Any]] = {}

    def fit(self, frame: pd.DataFrame) -> "HierarchicalBackoff":
        if TARGET_COL not in frame:
            raise ValueError(f"training frame is missing {TARGET_COL}")
        train = add_context_columns(frame)
        self.global_rate = float(train[TARGET_COL].mean())
        self.tables = {}
        for spec in self.specs:
            parent = _statistics(train, spec.parent_keys)
            parent["posterior"] = (
                parent["sum"] + self.alpha_parent * self.global_rate
            ) / (parent["count"] + self.alpha_parent)
            leaf = _statistics(train, spec.leaf_keys)
            self.tables[spec.name] = {
                "parent": parent,
                "leaf": leaf,
            }
        return self

    @staticmethod
    def _lookup(
        values: pd.DataFrame,
        table: pd.DataFrame,
        keys: tuple[str, ...],
        columns: list[str],
    ) -> pd.DataFrame:
        order = pd.Series(np.arange(len(values)), name="__order")
        left = values[list(keys)].copy()
        left["__order"] = order
        merged = left.merge(
            table[list(keys) + columns],
            on=list(keys),
            how="left",
            sort=False,
            validate="many_to_one",
        ).sort_values("__order", kind="stable")
        return merged[columns].reset_index(drop=True)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.global_rate is None or not self.tables:
            raise RuntimeError("HierarchicalBackoff must be fit before transform")
        values = add_context_columns(frame)
        output = pd.DataFrame(index=np.arange(len(values)))
        combined = np.zeros(len(values), dtype=np.float64)
        total_weight = 0.0
        for spec in self.specs:
            tables = self.tables[spec.name]
            parent_lookup = self._lookup(
                values,
                tables["parent"],
                spec.parent_keys,
                ["posterior", "count"],
            )
            parent_rate = parent_lookup["posterior"].fillna(self.global_rate).to_numpy(float)
            parent_count = parent_lookup["count"].fillna(0).to_numpy(float)
            leaf_lookup = self._lookup(
                values,
                tables["leaf"],
                spec.leaf_keys,
                ["sum", "count"],
            )
            leaf_sum = leaf_lookup["sum"].fillna(0).to_numpy(float)
            leaf_count = leaf_lookup["count"].fillna(0).to_numpy(float)
            posterior = (leaf_sum + self.alpha_leaf * parent_rate) / (
                leaf_count + self.alpha_leaf
            )
            output[f"{spec.name}_rate"] = posterior.astype("float32")
            output[f"{spec.name}_log_n"] = np.log1p(leaf_count).astype("float32")
            output[f"{spec.name}_posterior_se"] = np.sqrt(
                posterior * (1.0 - posterior) / (leaf_count + self.alpha_leaf + 1.0)
            ).astype("float32")
            output[f"{spec.name}_backoff"] = np.select(
                [leaf_count > 0, parent_count > 0],
                [0, 1],
                default=2,
            ).astype("int8")
            combined += spec.weight * posterior
            total_weight += spec.weight
        output["hierarchical_context_rate"] = (combined / total_weight).astype("float32")
        return output

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame)["hierarchical_context_rate"].to_numpy(dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        if self.global_rate is None:
            raise RuntimeError("cannot serialize an unfitted backoff")
        return {
            "alpha_leaf": self.alpha_leaf,
            "alpha_parent": self.alpha_parent,
            "global_rate": self.global_rate,
            "specs": [
                {
                    "name": spec.name,
                    "leaf_keys": list(spec.leaf_keys),
                    "parent_keys": list(spec.parent_keys),
                    "weight": spec.weight,
                }
                for spec in self.specs
            ],
        }
