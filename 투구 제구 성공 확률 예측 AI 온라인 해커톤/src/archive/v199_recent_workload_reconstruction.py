"""Audit row-local reconstruction of recent-game pitcher workloads.

The official table exposes success and middle rates over the previous one,
three, and five pitcher appearances, but omits their pitch counts.  Because
both rates share a denominator and are rounded to six decimal places, their
minimum common integer denominator is often recoverable from a single row.

This module first checks that interpretation against chronological pitcher
game blocks in official ``train.csv``.  It does not read test data, pool query
rows, or use audit-row labels to construct an inference feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "V199_RECENT_WORKLOAD_RECONSTRUCTION_V1"
RATE_DECIMALS = 6
ROUNDING_TOLERANCE = 0.5 * 10.0 ** (-RATE_DECIMALS) + 1e-12
HORIZONS = (1, 3, 5)
MAX_DENOMINATOR = {1: 240, 3: 600, 5: 1000}
BASE_COLUMNS = (
    "row_id",
    "season",
    "inning",
    "top_bottom",
    "pitcher_id",
    "control_success",
)


def rate_columns(horizon: int) -> tuple[str, str]:
    return (
        f"asof_pitcher_prev{horizon}_game_success_rate",
        f"asof_pitcher_prev{horizon}_game_middle_rate",
    )


def infer_min_common_denominator(
    success_rate: float,
    middle_rate: float,
    *,
    max_denominator: int,
    tolerance: float = ROUNDING_TOLERANCE,
) -> tuple[int, float, bool]:
    """Return the smallest denominator compatible with two rounded rates.

    The result is deliberately a lower bound.  If both event counts share a
    factor with the true pitch count, the true denominator is not identifiable
    from the rates alone.  ``fit`` distinguishes a valid rounded-rational match
    from the closest fallback returned when the configured upper bound is too
    small.
    """

    rates = np.asarray([success_rate, middle_rate], dtype=np.float64)
    if not np.all(np.isfinite(rates)):
        return 0, float("nan"), False
    if np.any((rates < 0.0) | (rates > 1.0)):
        return 0, float("nan"), False
    denominators = np.arange(1, int(max_denominator) + 1, dtype=np.float64)
    reconstructed = np.rint(denominators[:, None] * rates[None, :])
    reconstructed /= denominators[:, None]
    error = np.max(np.abs(reconstructed - rates[None, :]), axis=1)
    compatible = np.flatnonzero(error <= float(tolerance))
    index = int(compatible[0]) if compatible.size else int(np.argmin(error))
    return index + 1, float(error[index]), bool(compatible.size)


def infer_denominators(
    success_rate: pd.Series,
    middle_rate: pd.Series,
    *,
    max_denominator: int,
) -> pd.DataFrame:
    """Infer denominators once per unique rate pair and map them to rows."""

    values = pd.DataFrame(
        {
            "success_rate": pd.to_numeric(success_rate, errors="coerce"),
            "middle_rate": pd.to_numeric(middle_rate, errors="coerce"),
        }
    )
    unique = values.drop_duplicates().reset_index(drop=True)
    inferred = [
        infer_min_common_denominator(
            row.success_rate,
            row.middle_rate,
            max_denominator=max_denominator,
        )
        for row in unique.itertuples(index=False)
    ]
    unique[["minimum_denominator", "fit_error", "rounded_rational_fit"]] = inferred
    unique["rounded_rational_fit"] = unique["rounded_rational_fit"].astype(bool)
    unique["uninformative_pair"] = (
        unique["success_rate"].isin([0.0, 1.0])
        & unique["middle_rate"].isin([0.0, 1.0])
    )
    mapped = values.merge(
        unique,
        on=["success_rate", "middle_rate"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    return mapped[
        [
            "minimum_denominator",
            "fit_error",
            "rounded_rational_fit",
            "uninformative_pair",
        ]
    ]


def attach_main_game_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the target-free chronological game boundary used elsewhere."""

    output = frame.copy()
    first_top = (
        pd.to_numeric(output["inning"], errors="coerce").eq(1)
        & output["top_bottom"].astype(str).eq("T")
    )
    previous_first_top = (
        pd.to_numeric(output["inning"], errors="coerce").shift().eq(1)
        & output["top_bottom"].shift().astype(str).eq("T")
    )
    starts = first_top & ~previous_first_top
    starts |= output["season"].ne(output["season"].shift())
    output["main_game_index"] = starts.cumsum().astype(np.int32)
    return output


def pitcher_game_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse chronological rows to one record per pitcher appearance."""

    required = set(BASE_COLUMNS)
    for horizon in HORIZONS:
        required.update(rate_columns(horizon))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    indexed = attach_main_game_index(frame)
    aggregations: dict[str, tuple[str, str]] = {
        "first_row_id": ("row_id", "first"),
        "pitches": ("control_success", "size"),
        "successes": ("control_success", "sum"),
    }
    for horizon in HORIZONS:
        success, middle = rate_columns(horizon)
        aggregations[f"prev{horizon}_success_rate"] = (success, "first")
        aggregations[f"prev{horizon}_middle_rate"] = (middle, "first")
        aggregations[f"prev{horizon}_success_nunique"] = (success, "nunique")
        aggregations[f"prev{horizon}_middle_nunique"] = (middle, "nunique")
    games = (
        indexed.groupby(
            ["season", "main_game_index", "pitcher_id"],
            sort=False,
            observed=True,
        )
        .agg(**aggregations)
        .reset_index()
    )
    games["successes"] = games["successes"].astype(np.int32)
    games["pitches"] = games["pitches"].astype(np.int32)
    return games


def attach_true_previous_windows(games: pd.DataFrame) -> pd.DataFrame:
    """Attach label-based train diagnostics; never deploy these columns."""

    output = games.copy()
    for horizon in HORIZONS:
        previous_pitches = output.groupby("pitcher_id", sort=False)["pitches"].transform(
            lambda values, h=horizon: values.shift(1).rolling(h, min_periods=1).sum()
        )
        previous_successes = output.groupby("pitcher_id", sort=False)[
            "successes"
        ].transform(
            lambda values, h=horizon: values.shift(1).rolling(h, min_periods=1).sum()
        )
        output[f"prev{horizon}_true_pitches"] = previous_pitches
        output[f"prev{horizon}_true_success_rate"] = np.divide(
            previous_successes,
            previous_pitches,
            out=np.full(len(output), np.nan, dtype=np.float64),
            where=previous_pitches.to_numpy(np.float64) > 0.0,
        )
    return output


def attach_inferred_workloads(games: pd.DataFrame) -> pd.DataFrame:
    output = games.copy()
    for horizon in HORIZONS:
        inferred = infer_denominators(
            output[f"prev{horizon}_success_rate"],
            output[f"prev{horizon}_middle_rate"],
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        for column in inferred:
            output[f"prev{horizon}_{column}"] = inferred[column].to_numpy()
    return output


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def horizon_summary(games: pd.DataFrame, horizon: int) -> dict[str, Any]:
    official_rate = games[f"prev{horizon}_success_rate"].to_numpy(np.float64)
    true_rate = games[f"prev{horizon}_true_success_rate"].to_numpy(np.float64)
    true_n = games[f"prev{horizon}_true_pitches"].to_numpy(np.float64)
    inferred_n = games[f"prev{horizon}_minimum_denominator"].to_numpy(np.float64)
    fit = games[f"prev{horizon}_rounded_rational_fit"].fillna(False).to_numpy(bool)
    uninformative = games[f"prev{horizon}_uninformative_pair"].fillna(True).to_numpy(bool)
    finite = np.isfinite(official_rate) & np.isfinite(true_rate) & np.isfinite(true_n)
    rate_match = finite & (np.abs(official_rate - true_rate) <= ROUNDING_TOLERANCE)
    comparable = rate_match & fit & ~uninformative & (inferred_n > 0.0)
    exact = comparable & np.equal(inferred_n, true_n)
    divisor = comparable & np.equal(np.mod(true_n, inferred_n), 0.0)
    ratio = np.divide(
        inferred_n,
        true_n,
        out=np.full(len(games), np.nan, dtype=np.float64),
        where=comparable & (true_n > 0.0),
    )
    return {
        "rows": int(len(games)),
        "official_non_missing": int(np.isfinite(official_rate).sum()),
        "rate_semantics_match": int(rate_match.sum()),
        "rate_semantics_match_fraction": _safe_fraction(int(rate_match.sum()), int(finite.sum())),
        "rounded_rational_fit_fraction": _safe_fraction(int((finite & fit).sum()), int(finite.sum())),
        "informative_comparable": int(comparable.sum()),
        "exact_denominator_fraction": _safe_fraction(int(exact.sum()), int(comparable.sum())),
        "divisor_or_exact_fraction": _safe_fraction(int(divisor.sum()), int(comparable.sum())),
        "median_inferred_over_true": float(np.nanmedian(ratio)) if comparable.any() else float("nan"),
        "p10_inferred_over_true": float(np.nanquantile(ratio, 0.10)) if comparable.any() else float("nan"),
        "true_pitch_count_p50": float(np.nanmedian(true_n[finite])) if finite.any() else float("nan"),
        "true_pitch_count_p99": float(np.nanquantile(true_n[finite], 0.99)) if finite.any() else float("nan"),
        "true_pitch_count_max": float(np.nanmax(true_n[finite])) if finite.any() else float("nan"),
        "within_game_success_rate_changed": int(
            games[f"prev{horizon}_success_nunique"].gt(1).sum()
        ),
        "within_game_middle_rate_changed": int(
            games[f"prev{horizon}_middle_nunique"].gt(1).sum()
        ),
    }


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "deployed_inference_is_row_local": True,
        "true_workload_is_diagnostic_only": True,
        "audit_labels_not_used_in_inferred_denominator": True,
    }


def run(train_csv: Path, output_dir: Path) -> dict[str, Any]:
    columns = list(BASE_COLUMNS)
    for horizon in HORIZONS:
        columns.extend(rate_columns(horizon))
    frame = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    games = attach_inferred_workloads(
        attach_true_previous_windows(pitcher_game_table(frame))
    )
    summaries = {str(horizon): horizon_summary(games, horizon) for horizon in HORIZONS}
    semantic_pass = all(
        item["rate_semantics_match_fraction"] >= 0.995 for item in summaries.values()
    )
    useful_pass = all(
        item["divisor_or_exact_fraction"] >= 0.99
        and item["median_inferred_over_true"] >= 0.50
        for item in summaries.values()
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "validated_signal" if semantic_pass and useful_pass else "diagnostic_reject",
        "pitcher_game_rows": int(len(games)),
        "horizons": summaries,
        "semantic_gate_passed": bool(semantic_pass),
        "usefulness_gate_passed": bool(useful_pass),
        "eligible_for_model_screen": bool(semantic_pass and useful_pass),
        "restrictions": restrictions(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    games.to_parquet(output_dir / "pitcher_game_workload_audit.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_csv, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
