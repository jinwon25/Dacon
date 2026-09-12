"""Audit a legal row-local estimate of batting order and times-through-order.

Chronological ``train.csv`` rows are used only to reconstruct target-free game
and plate-appearance states.  For an audit season, all lookup tables are built
from completed seasons strictly before it.  At inference the estimator needs
only fields on the current row; evaluation row order and neighbouring rows are
never consumed.

The context lookup estimates the offensive plate-appearance ordinal from the
inning, batting-team score, outs and occupied bases.  A batter's prior typical
lineup slot then snaps that continuous estimate onto a legal 9-slot batting
cycle.  This supplies an explicit times-through-order proxy that is absent
from the competition table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROTOCOL = "V307_BATTING_ORDER_TTO_AUDIT_V1"
AUDIT_YEARS = (2022, 2023, 2024)
MAX_CYCLE = 7
COLUMNS = (
    "row_id",
    "season",
    "game_type",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "num_runners_on",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "batter_id",
    "batter_team_id",
)


def attach_batting_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach diagnostic game, PA ordinal, lineup slot and TTO to train rows."""

    output = frame.copy()
    first_top = (
        pd.to_numeric(output["inning"], errors="coerce").eq(1)
        & output["top_bottom"].astype(str).eq("T")
    )
    starts = first_top & ~first_top.shift(fill_value=False)
    starts |= output["season"].ne(output["season"].shift())
    output["_game_id"] = starts.cumsum().astype(np.int32)

    keys = ["season", "_game_id", "batter_team_id"]
    grouped = output.groupby(keys, sort=False, observed=True)
    first_team_row = grouped.cumcount().eq(0)
    previous_batter = grouped["batter_id"].shift()
    previous_balls = grouped["balls_before"].shift()
    previous_strikes = grouped["strikes_before"].shift()
    zero_count = output["balls_before"].eq(0) & output["strikes_before"].eq(0)
    previous_zero = previous_balls.eq(0) & previous_strikes.eq(0)
    # A batter change at a non-zero count is a mid-PA substitution, not a new
    # trip through the order.  Consecutive 0-0 rows are protected for rare
    # no-count pitch records.
    new_pa = first_team_row | (
        zero_count
        & (output["batter_id"].ne(previous_batter) | ~previous_zero)
    )
    output["_pa_index"] = (
        new_pa.astype(np.int32)
        .groupby([output[key] for key in keys], sort=False)
        .cumsum()
        .astype(np.int16)
    )
    output["_lineup_slot"] = ((output["_pa_index"] - 1) % 9 + 1).astype(np.int8)
    output["_true_tto"] = ((output["_pa_index"] - 1) // 9 + 1).astype(np.int8)
    output["_offense_runs"] = np.where(
        output["top_bottom"].astype(str).eq("T"),
        pd.to_numeric(output["run_top_before"], errors="coerce"),
        pd.to_numeric(output["run_bot_before"], errors="coerce"),
    )
    return output


def _first_pa_rows(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "_game_id", "batter_team_id", "_pa_index"]
    return rows.drop_duplicates(keys, keep="first").copy()


def _context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inning_key": np.clip(
                pd.to_numeric(frame["inning"], errors="coerce").fillna(1), 1, 12
            ).astype(np.int8),
            "runs_key": np.clip(
                pd.to_numeric(frame["_offense_runs"], errors="coerce").fillna(0),
                0,
                10,
            ).astype(np.int8),
            "outs_key": np.clip(
                pd.to_numeric(frame["outs_before"], errors="coerce").fillna(0), 0, 2
            ).astype(np.int8),
            "runners_key": np.clip(
                pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0),
                0,
                3,
            ).astype(np.int8),
            "half_key": frame["top_bottom"].astype(str).to_numpy(),
        },
        index=frame.index,
    )


def _aggregate_context(
    history_pa: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    grouped = history_pa.groupby(keys, observed=True, sort=False)["_pa_index"]
    table = grouped.agg(pa_expected="median", context_n="size")
    table["p_tto2"] = grouped.apply(lambda values: float((values >= 10).mean()))
    table["p_tto3"] = grouped.apply(lambda values: float((values >= 19).mean()))
    return table


def _map_context_level(
    query: pd.DataFrame, table: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(query[keys]) if len(keys) > 1 else pd.Index(
        query[keys[0]], name=keys[0]
    )
    mapped = table.reindex(index)
    mapped.index = query.index
    return mapped


def _fill_context_hierarchy(
    history_pa: pd.DataFrame, query: pd.DataFrame
) -> pd.DataFrame:
    levels = (
        ["inning_key", "runs_key", "outs_key", "runners_key", "half_key"],
        ["inning_key", "runs_key", "outs_key", "runners_key"],
        ["inning_key", "runs_key", "outs_key"],
        ["inning_key", "runs_key"],
        ["inning_key"],
    )
    result = pd.DataFrame(
        {
            "pa_expected_context": np.nan,
            "context_p_tto2": np.nan,
            "context_p_tto3": np.nan,
            "context_n": 0.0,
            "context_level": 0.0,
        },
        index=query.index,
    )
    for level_number, keys in enumerate(levels, start=1):
        missing = result["pa_expected_context"].isna()
        if not missing.any():
            break
        table = _aggregate_context(history_pa, list(keys))
        mapped = _map_context_level(query.loc[missing], table, list(keys))
        available = mapped["pa_expected"].notna()
        selected = mapped.index[available]
        result.loc[selected, "pa_expected_context"] = mapped.loc[
            selected, "pa_expected"
        ]
        result.loc[selected, "context_p_tto2"] = mapped.loc[selected, "p_tto2"]
        result.loc[selected, "context_p_tto3"] = mapped.loc[selected, "p_tto3"]
        result.loc[selected, "context_n"] = mapped.loc[selected, "context_n"]
        result.loc[selected, "context_level"] = float(level_number)
    return result


def _mode_lookup(
    history: pd.DataFrame, query: pd.DataFrame, keys: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    group_keys = list(keys)
    counts = (
        history.groupby(group_keys + ["_lineup_slot"], observed=True, sort=False)
        .size()
        .rename("slot_n")
        .reset_index()
        .sort_values(group_keys + ["slot_n", "_lineup_slot"],
                     ascending=[True] * len(group_keys) + [False, True])
        .drop_duplicates(group_keys, keep="first")
        .set_index(group_keys)
    )
    if len(group_keys) == 1:
        index = pd.Index(query[group_keys[0]], name=group_keys[0])
    else:
        index = pd.MultiIndex.from_frame(query[group_keys])
    mapped = counts.reindex(index)
    return (
        mapped["_lineup_slot"].to_numpy(np.float64),
        mapped["slot_n"].fillna(0).to_numpy(np.float64),
    )


def _snap_to_slot(pa_expected: np.ndarray, slot: np.ndarray) -> np.ndarray:
    result = np.asarray(pa_expected, dtype=np.float64).copy()
    valid = np.isfinite(result) & np.isfinite(slot)
    if not valid.any():
        return result
    cycles = np.arange(MAX_CYCLE, dtype=np.float64)
    candidates = slot[valid, None] + 9.0 * cycles[None, :]
    nearest = np.argmin(np.abs(candidates - result[valid, None]), axis=1)
    result[valid] = candidates[np.arange(valid.sum()), nearest]
    return result


def build_strict_batting_features(rows: pd.DataFrame, year: int) -> pd.DataFrame:
    """Build deployable features for regular-season rows in ``year``."""

    history = rows.loc[
        rows["season"].lt(year) & rows["game_type"].astype(str).eq("R")
    ]
    audit = rows.loc[
        rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    ]
    if history.empty or audit.empty:
        raise ValueError(f"missing strict history or audit rows for {year}")
    history_pa = _first_pa_rows(history)
    history_context = _context_columns(history_pa)
    history_pa = history_pa.join(history_context)
    query_context = _context_columns(audit).reset_index(drop=True)
    context = _fill_context_hierarchy(
        history_pa.reset_index(drop=True), query_context
    )

    # One observation per batter per game prevents long plate appearances from
    # receiving extra weight in the typical-slot lookup.
    history_batter_games = history_pa.drop_duplicates(
        ["season", "_game_id", "batter_team_id", "batter_id"], keep="first"
    )
    query_ids = audit[["batter_id", "batter_team_id"]].reset_index(drop=True)
    slot, slot_n = _mode_lookup(
        history_batter_games, query_ids, ["batter_id", "batter_team_id"]
    )
    missing = ~np.isfinite(slot)
    if missing.any():
        fallback_slot, fallback_n = _mode_lookup(
            history_batter_games, query_ids.loc[missing], ["batter_id"]
        )
        slot[missing] = fallback_slot
        slot_n[missing] = fallback_n

    expected = context["pa_expected_context"].to_numpy(np.float64)
    snapped = _snap_to_slot(expected, slot)
    return pd.DataFrame(
        {
            "pa_expected_context": expected,
            "pa_expected_slot_snap": snapped,
            "predicted_lineup_slot": slot,
            "lineup_history_games": slot_n,
            "context_p_tto2": context["context_p_tto2"].to_numpy(np.float64),
            "context_p_tto3": context["context_p_tto3"].to_numpy(np.float64),
            "predicted_tto": np.floor((snapped - 1.0) / 9.0) + 1.0,
            "context_n": context["context_n"].to_numpy(np.float64),
            "context_level": context["context_level"].to_numpy(np.float64),
            "lineup_covered": np.isfinite(slot).astype(np.float64),
        }
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def audit_metrics(
    rows: pd.DataFrame, features: pd.DataFrame, year: int
) -> dict[str, Any]:
    mask = rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    audit = rows.loc[mask]
    truth_pa = audit["_pa_index"].to_numpy(np.float64)
    truth_slot = audit["_lineup_slot"].to_numpy(np.float64)
    truth_tto = audit["_true_tto"].to_numpy(np.float64)
    context = features["pa_expected_context"].to_numpy(np.float64)
    snapped = features["pa_expected_slot_snap"].to_numpy(np.float64)
    slot = features["predicted_lineup_slot"].to_numpy(np.float64)
    covered = np.isfinite(slot) & np.isfinite(snapped)
    circular_slot_error = np.minimum(
        np.abs(slot[covered] - truth_slot[covered]),
        9.0 - np.abs(slot[covered] - truth_slot[covered]),
    )
    result: dict[str, Any] = {
        "rows": int(len(audit)),
        "lineup_coverage": float(covered.mean()),
        "lineup_exact_accuracy": float(np.mean(circular_slot_error == 0)),
        "lineup_within_one_accuracy": float(np.mean(circular_slot_error <= 1)),
        "context_pa_correlation": _safe_corr(truth_pa, context),
        "context_pa_mae": float(np.nanmean(np.abs(truth_pa - context))),
        "snapped_pa_correlation": _safe_corr(truth_pa[covered], snapped[covered]),
        "snapped_pa_mae": float(np.mean(np.abs(truth_pa[covered] - snapped[covered]))),
        "predicted_tto_accuracy": float(np.mean(
            features.loc[covered, "predicted_tto"].to_numpy(np.float64)
            == truth_tto[covered]
        )),
    }
    for threshold, probability in ((2, "context_p_tto2"), (3, "context_p_tto3")):
        target = truth_tto >= threshold
        scores = features[probability].to_numpy(np.float64)
        valid = np.isfinite(scores)
        result[f"tto_ge{threshold}_auc"] = float(
            roc_auc_score(target[valid], scores[valid])
        ) if np.unique(target[valid]).size == 2 else float("nan")
        result[f"tto_ge{threshold}_prevalence"] = float(target.mean())
    return result


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "control_success_read": False,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_lookup": True,
        "train_order_used_only_for_target_free_lookup_construction": True,
    }


def run(train_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, usecols=list(COLUMNS), low_memory=False)
    rows = attach_batting_state(raw)
    metrics: dict[str, Any] = {}
    for year in AUDIT_YEARS:
        features = build_strict_batting_features(rows, year)
        metrics[str(year)] = audit_metrics(rows, features, year)
        features.to_parquet(output_dir / f"batting_features_{year}.parquet", index=False)
    eligible = bool(all(
        item["lineup_coverage"] >= 0.70
        and item["snapped_pa_correlation"] >= 0.75
        and item["tto_ge2_auc"] >= 0.80
        for item in metrics.values()
    ))
    result = {
        "protocol": PROTOCOL,
        "status": "signal_present" if eligible else "diagnostic_reject",
        "metrics": metrics,
        "eligible_for_residual_screen": eligible,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
