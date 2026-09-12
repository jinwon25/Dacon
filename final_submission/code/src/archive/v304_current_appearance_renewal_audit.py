"""Audit a row-local renewal estimate of current-appearance pitch count.

``train.csv`` is chronological, which lets us reconstruct pitcher appearances
for *training diagnostics and lookup construction*.  Evaluation-row order is
never used.  At inference, the only live input is the current row's official
career pitch counter.  A precomputed pitcher-specific distribution of prior
appearance lengths is treated as a renewal process, yielding the posterior
age (pitches already thrown in the current appearance) at that counter value.

This module is target-free: ``control_success`` is not read.  It answers the
prerequisite question for a later model experiment -- whether the otherwise
missing fatigue clock is recoverable from legal, row-local information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROTOCOL = "V304_CURRENT_APPEARANCE_RENEWAL_AUDIT_V1"
MAX_LENGTH = 180
PRIOR_STRENGTH = 8.0
AUDIT_YEARS = (2022, 2023, 2024)
COLUMNS = (
    "row_id",
    "season",
    "game_type",
    "inning",
    "top_bottom",
    "pitcher_id",
    "asof_pitcher_n",
)


def attach_appearance_state(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rows with diagnostic appearance age and an appearance table."""

    output = frame.copy()
    first_top = output["inning"].eq(1) & output["top_bottom"].astype(str).eq("T")
    starts = first_top & ~first_top.shift(fill_value=False)
    starts |= output["season"].ne(output["season"].shift())
    output["_game_id"] = starts.cumsum().astype(np.int32)
    keys = ["season", "_game_id", "pitcher_id"]
    output["_appearance_age"] = (
        output.groupby(keys, sort=False, observed=True).cumcount().astype(np.int16)
    )
    appearances = (
        output.groupby(keys, sort=False, observed=True)
        .agg(
            game_type=("game_type", "first"),
            first_inning=("inning", "first"),
            last_inning=("inning", "last"),
            pitches=("row_id", "size"),
        )
        .reset_index()
    )
    return output, appearances


def _length_pmf(lengths: np.ndarray, global_pmf: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(lengths, dtype=np.int64), 1, MAX_LENGTH)
    counts = np.bincount(clipped, minlength=MAX_LENGTH + 1).astype(np.float64)
    counts[0] = 0.0
    return (counts + PRIOR_STRENGTH * global_pmf) / (
        counts.sum() + PRIOR_STRENGTH
    )


def renewal_posterior_table(
    pmf: np.ndarray, maximum_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean/SD and two fatigue probabilities for each total count."""

    probability = np.asarray(pmf, dtype=np.float64)
    if probability.shape != (MAX_LENGTH + 1,) or probability[0] != 0.0:
        raise ValueError("unexpected appearance-length PMF")
    total_max = max(int(maximum_total), 0)
    renewal = np.zeros(total_max + 1, dtype=np.float64)
    renewal[0] = 1.0
    support = np.flatnonzero(probability > 0.0)
    for total in range(1, total_max + 1):
        valid = support[support <= total]
        if len(valid):
            renewal[total] = float(np.dot(probability[valid], renewal[total - valid]))

    # P(L > age): the current appearance must survive beyond pitches already
    # thrown.  Age zero is the first pitch of an appearance.
    survival = np.asarray(
        [probability[age + 1 :].sum() for age in range(MAX_LENGTH)],
        dtype=np.float64,
    )
    mean = np.zeros(total_max + 1, dtype=np.float64)
    sd = np.zeros(total_max + 1, dtype=np.float64)
    p20 = np.zeros(total_max + 1, dtype=np.float64)
    p60 = np.zeros(total_max + 1, dtype=np.float64)
    for total in range(total_max + 1):
        ages = np.arange(min(total, MAX_LENGTH - 1) + 1, dtype=np.int64)
        weights = renewal[total - ages] * survival[ages]
        denominator = float(weights.sum())
        if denominator <= 0.0:
            continue
        weights /= denominator
        center = float(np.dot(ages, weights))
        mean[total] = center
        sd[total] = float(np.sqrt(np.dot(np.square(ages - center), weights)))
        p20[total] = float(weights[ages >= 20].sum())
        p60[total] = float(weights[ages >= 60].sum())
    return mean, sd, p20, p60


def _global_pmf(appearances: pd.DataFrame) -> np.ndarray:
    lengths = np.clip(
        appearances["pitches"].to_numpy(np.int64), 1, MAX_LENGTH
    )
    counts = np.bincount(lengths, minlength=MAX_LENGTH + 1).astype(np.float64)
    counts[0] = 0.0
    return counts / counts.sum()


def build_strict_renewal_features(
    rows: pd.DataFrame, appearances: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Build features for ``year`` using completed seasons strictly before it."""

    audit = rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    audit_rows = rows.loc[audit]
    history_rows = rows.loc[rows["season"].lt(year)]
    history_apps = appearances.loc[
        appearances["season"].lt(year)
        & appearances["game_type"].astype(str).eq("R")
    ]
    if history_rows.empty or history_apps.empty:
        raise ValueError(f"no history available for {year}")

    # The last row exposes the pre-pitch counter.  Adding one gives the exact
    # number of completed pitches at the history boundary without a label.
    anchors = (
        history_rows.groupby("pitcher_id", sort=False)["asof_pitcher_n"].last()
        + 1.0
    )
    global_pmf = _global_pmf(history_apps)
    global_lengths = history_apps["pitches"].to_numpy(np.int64)
    pitcher_lengths = {
        int(pitcher): group["pitches"].to_numpy(np.int64)
        for pitcher, group in history_apps.groupby("pitcher_id", sort=False)
    }

    pitcher = audit_rows["pitcher_id"].to_numpy(np.int64)
    counter = pd.to_numeric(
        audit_rows["asof_pitcher_n"], errors="coerce"
    ).to_numpy(np.float64)
    anchor = pd.Series(pitcher).map(anchors).to_numpy(np.float64)
    delta = np.maximum(np.nan_to_num(counter - anchor, nan=0.0), 0.0).astype(np.int64)
    covered = np.isfinite(anchor)
    output = pd.DataFrame(
        {
            "renewal_expected_age": np.nan,
            "renewal_age_sd": np.nan,
            "renewal_p_age_ge20": np.nan,
            "renewal_p_age_ge60": np.nan,
            "renewal_history_appearances": 0.0,
            "renewal_history_median_length": float(np.median(global_lengths)),
            "renewal_counter_delta": delta.astype(np.float64),
            "renewal_covered": covered.astype(np.float64),
        },
        index=np.arange(len(audit_rows)),
    )
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for player in np.unique(pitcher[covered]):
        selected = pitcher == player
        lengths = pitcher_lengths.get(int(player), global_lengths)
        pmf = _length_pmf(lengths, global_pmf)
        table = renewal_posterior_table(pmf, int(delta[selected].max()))
        output.loc[selected, "renewal_expected_age"] = table[0][delta[selected]]
        output.loc[selected, "renewal_age_sd"] = table[1][delta[selected]]
        output.loc[selected, "renewal_p_age_ge20"] = table[2][delta[selected]]
        output.loc[selected, "renewal_p_age_ge60"] = table[3][delta[selected]]
        output.loc[selected, "renewal_history_appearances"] = float(len(lengths))
        output.loc[selected, "renewal_history_median_length"] = float(
            np.median(lengths)
        )
        cache[int(player)] = table
    return output


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def audit_metrics(
    rows: pd.DataFrame, features: pd.DataFrame, year: int
) -> dict[str, Any]:
    audit = rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    truth = rows.loc[audit, "_appearance_age"].to_numpy(np.float64)
    expected = features["renewal_expected_age"].to_numpy(np.float64)
    covered = features["renewal_covered"].to_numpy(bool) & np.isfinite(expected)
    true_heavy20 = truth >= 20.0
    true_heavy60 = truth >= 60.0
    result: dict[str, Any] = {
        "rows": int(len(truth)),
        "covered_rows": int(covered.sum()),
        "coverage": float(covered.mean()),
        "age_correlation": _safe_corr(truth[covered], expected[covered]),
        "age_mae": float(np.mean(np.abs(truth[covered] - expected[covered]))),
        "zero_age_mae": float(np.mean(np.abs(truth[covered]))),
        "truth_age_mean": float(np.mean(truth[covered])),
        "prediction_age_mean": float(np.mean(expected[covered])),
    }
    for threshold, target, column in (
        (20, true_heavy20, "renewal_p_age_ge20"),
        (60, true_heavy60, "renewal_p_age_ge60"),
    ):
        if np.unique(target[covered]).size == 2:
            result[f"age_ge{threshold}_auc"] = float(
                roc_auc_score(target[covered], features.loc[covered, column])
            )
        else:
            result[f"age_ge{threshold}_auc"] = float("nan")
        result[f"age_ge{threshold}_prevalence"] = float(target[covered].mean())
    return result


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "control_success_read": False,
        "row_id_used_only_to_validate_train_chronology": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_appearance_lookup": True,
    }


def run(train_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(train_csv, usecols=list(COLUMNS), low_memory=False)
    numeric_id = pd.to_numeric(
        frame["row_id"].astype(str).str.rsplit("_", n=1).str[-1], errors="coerce"
    ).to_numpy(np.float64)
    chronological = bool(
        np.isfinite(numeric_id).all() and np.all(np.diff(numeric_id) == 1.0)
    )
    if not chronological:
        raise ValueError("train row_id is not a contiguous chronological key")
    rows, appearances = attach_appearance_state(frame)
    metrics: dict[str, Any] = {}
    for year in AUDIT_YEARS:
        features = build_strict_renewal_features(rows, appearances, year)
        metrics[str(year)] = audit_metrics(rows, features, year)
        features.to_parquet(output_dir / f"renewal_features_{year}.parquet", index=False)
    result = {
        "protocol": PROTOCOL,
        "status": "signal_present" if all(
            item["age_correlation"] > 0.10 and item["age_ge20_auc"] > 0.55
            for item in metrics.values()
        ) else "diagnostic_reject",
        "train_rows_chronological": chronological,
        "appearance_rows": int(len(appearances)),
        "metrics": metrics,
        "eligible_for_residual_screen": bool(all(
            item["age_correlation"] > 0.10 and item["age_ge20_auc"] > 0.55
            for item in metrics.values()
        )),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.train_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
