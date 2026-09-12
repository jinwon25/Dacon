"""Standalone train-only failure-prior bank used by the v116 public probe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "V117_FAILURE_PRIOR_BANK_V1"
COMPONENTS = ("reverse", "middle", "wayoff")


def _wide_counts(
    frame: pd.DataFrame, keys: list[str], pitch_types: list[str]
) -> pd.DataFrame:
    table = (
        frame.groupby([*keys, "pitch_type_fine"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pitch_types, fill_value=0)
    )
    table.columns = [f"n__{name}" for name in pitch_types]
    return table.reset_index()


def build_failure_bank(
    history: pd.DataFrame,
    pitch_types: list[str],
    *,
    outcome_shrink: float,
    selection_shrink: float,
) -> dict[str, Any]:
    """Precompute all historical tables required for row-local inference."""
    if history.empty:
        raise ValueError("failure-prior history is empty")
    required = {
        "season", "pitcher_id", "batter_hand", "count_state", "pitch_type_fine",
        *[f"failure__{name}" for name in COMPONENTS],
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"failure-prior history columns missing: {missing}")
    work = history.reset_index(drop=True).copy()

    global_mix_series = work["pitch_type_fine"].value_counts(normalize=True)
    global_mix = {
        name: float(global_mix_series.get(name, 0.0)) for name in pitch_types
    }
    overall = _wide_counts(work, ["pitcher_id"], pitch_types)
    count_columns = [f"n__{name}" for name in pitch_types]
    total = overall[count_columns].sum(axis=1).to_numpy(float)
    for name in pitch_types:
        overall[f"mix__{name}"] = (
            overall[f"n__{name}"].to_numpy(float)
            + float(selection_shrink) * global_mix[name]
        ) / (total + float(selection_shrink))

    keys = ["pitcher_id", "batter_hand", "count_state"]
    conditional = _wide_counts(work, keys, pitch_types)
    conditional_total = conditional[count_columns].sum(axis=1).to_numpy(float)
    conditional = conditional.merge(
        overall[["pitcher_id", *[f"mix__{name}" for name in pitch_types]]],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )
    for name in pitch_types:
        backing = conditional[f"mix__{name}"].fillna(global_mix[name])
        conditional[f"mix__{name}"] = (
            conditional[f"n__{name}"].to_numpy(float)
            + float(selection_shrink) * backing.to_numpy(float)
        ) / (conditional_total + float(selection_shrink))
    conditional = conditional[keys + [f"mix__{name}" for name in pitch_types]]

    outcomes: pd.DataFrame | None = None
    marginals: pd.DataFrame | None = None
    global_relative: dict[str, dict[str, float]] = {}
    for component in COMPONENTS:
        label = f"failure__{component}"
        relative = work[label].to_numpy(float) - work.groupby(
            "season", observed=True
        )[label].transform("mean").to_numpy(float)
        local = work[["pitcher_id", "pitch_type_fine"]].copy()
        local["relative"] = relative
        grouped = local.groupby("pitch_type_fine", observed=True)["relative"].agg(
            ["sum", "count"]
        )
        global_relative[component] = {
            name: (
                float(grouped.loc[name, "sum"] / grouped.loc[name, "count"])
                if name in grouped.index and grouped.loc[name, "count"] > 0
                else 0.0
            )
            for name in pitch_types
        }
        outcome = (
            local.groupby(["pitcher_id", "pitch_type_fine"], observed=True)["relative"]
            .agg(["sum", "count"])
            .reset_index()
        )
        outcome["value"] = [
            (float(total_value) + float(outcome_shrink) * global_relative[component][str(kind)])
            / (float(count) + float(outcome_shrink))
            for total_value, count, kind in zip(
                outcome["sum"], outcome["count"], outcome["pitch_type_fine"]
            )
        ]
        outcome = (
            outcome.pivot(index="pitcher_id", columns="pitch_type_fine", values="value")
            .reindex(columns=pitch_types)
        )
        outcome.columns = [f"{component}__value__{name}" for name in pitch_types]
        outcome = outcome.reset_index()
        outcomes = outcome if outcomes is None else outcomes.merge(
            outcome, on="pitcher_id", how="outer", validate="one_to_one"
        )

        value_columns = [f"{component}__value__{name}" for name in pitch_types]
        marginal = overall[["pitcher_id", *[f"mix__{name}" for name in pitch_types]]].merge(
            outcome, on="pitcher_id", how="left", validate="one_to_one"
        )
        marginal[f"{component}__expected"] = sum(
            marginal[f"mix__{name}"].fillna(0.0)
            * marginal[f"{component}__value__{name}"].fillna(
                global_relative[component][name]
            )
            for name in pitch_types
        )
        marginal = marginal[["pitcher_id", f"{component}__expected"]]
        marginals = marginal if marginals is None else marginals.merge(
            marginal, on="pitcher_id", how="outer", validate="one_to_one"
        )
        if len(value_columns) != len(pitch_types):
            raise AssertionError("unexpected failure-prior outcome shape")

    if outcomes is None or marginals is None:
        raise AssertionError("failure-prior component tables were not built")
    return {
        "protocol": PROTOCOL,
        "pitch_types": list(pitch_types),
        "global_mix": global_mix,
        "global_relative": global_relative,
        "conditional": conditional,
        "outcomes": outcomes,
        "marginals": marginals,
        "outcome_shrink": float(outcome_shrink),
        "selection_shrink": float(selection_shrink),
        "history_rows": int(len(work)),
        "row_local_inference": True,
        "test_aggregate_used": False,
    }


def predict_failure_components(rows: pd.DataFrame, bank: dict[str, Any]) -> np.ndarray:
    """Return reverse/middle/wayoff deltas using only fixed historical tables."""
    if bank.get("protocol") != PROTOCOL:
        raise ValueError("unexpected failure-prior bank protocol")
    pitch_types = [str(value) for value in bank["pitch_types"]]
    keys = ["pitcher_id", "batter_hand", "count_state"]
    query = rows[keys].reset_index(drop=True).copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(
        bank["conditional"], on=keys, how="left", sort=False, validate="many_to_one"
    )
    query = query.merge(
        bank["outcomes"], on="pitcher_id", how="left", sort=False,
        validate="many_to_one",
    )
    query = query.merge(
        bank["marginals"], on="pitcher_id", how="left", sort=False,
        validate="many_to_one",
    )
    output = np.zeros((len(query), len(COMPONENTS)), dtype=np.float64)
    for index, component in enumerate(COMPONENTS):
        expected = np.zeros(len(query), dtype=np.float64)
        for name in pitch_types:
            mix = query[f"mix__{name}"].fillna(float(bank["global_mix"][name]))
            value = query[f"{component}__value__{name}"].fillna(
                float(bank["global_relative"][component][name])
            )
            expected += mix.to_numpy(float) * value.to_numpy(float)
        output[:, index] = expected - query[f"{component}__expected"].fillna(
            0.0
        ).to_numpy(float)
    order = np.argsort(query["_order"].to_numpy())
    return output[order]
