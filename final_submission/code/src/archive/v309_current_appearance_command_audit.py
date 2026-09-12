"""Audit row-local recovery of command within the current appearance.

The official row provides an exact pre-pitch career count/rate and frozen
success/middle rates for the previous 1/3/5 appearances.  Strictly prior
season anchors turn the career values into current-season totals.  v304's
renewal clock estimates how many of those pitches belong to the current
appearance; rational recent-rate denominators estimate how much belongs to
the completed appearances immediately before it.  Their difference is a
row-local estimate of current-appearance success, unavailable as a raw field.

Chronological labels are used only to score reconstruction quality.  They do
not enter the deployable features for an audit season.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.archive.v298_recent_rate_denominator_signal import (
    MAX_DENOMINATOR,
    WINDOWS,
    decode_shared_denominator,
)
from src.archive.v304_current_appearance_renewal_audit import (
    attach_appearance_state,
    build_strict_renewal_features,
)
from src.archive.v306_completed_success_anchor_audit import (
    completed_success_anchors,
)


PROTOCOL = "V309_CURRENT_APPEARANCE_COMMAND_AUDIT_V1"
TARGET = "control_success"
AUDIT_YEARS = (2022, 2023, 2024)


def attach_appearance_success(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, appearances = attach_appearance_state(frame)
    keys = ["season", "_game_id", "pitcher_id"]
    cumulative = rows.groupby(keys, sort=False, observed=True)[TARGET].cumsum()
    rows["_appearance_success_before"] = (
        cumulative.to_numpy(np.float64) - rows[TARGET].to_numpy(np.float64)
    )
    return rows, appearances


def refine_denominator(
    minimum: np.ndarray,
    success_rate: np.ndarray,
    middle_rate: np.ndarray,
    expected: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Choose the compatible multiple nearest a prior workload expectation."""

    base = np.asarray(minimum, dtype=np.float64)
    success = np.asarray(success_rate, dtype=np.float64)
    middle = np.asarray(middle_rate, dtype=np.float64)
    expected_n = np.asarray(expected, dtype=np.float64)
    multiples = np.arange(1, 13, dtype=np.float64)
    candidates = base[:, None] * multiples[None, :]
    valid = np.isfinite(candidates) & (candidates >= 1.0) & (candidates <= maximum)
    success_error = np.abs(candidates * success[:, None] - np.rint(
        candidates * success[:, None]
    ))
    middle_error = np.abs(candidates * middle[:, None] - np.rint(
        candidates * middle[:, None]
    ))
    tolerance = 5.05e-7 * candidates + 1e-9
    valid &= np.maximum(success_error, middle_error) <= tolerance
    distance = np.abs(candidates - expected_n[:, None])
    distance[~valid] = np.inf
    choice = np.argmin(distance, axis=1)
    refined = candidates[np.arange(len(base)), choice]
    no_choice = ~np.isfinite(distance[np.arange(len(base)), choice])
    refined[no_choice] = base[no_choice]
    return refined


def build_current_appearance_features(
    rows: pd.DataFrame, appearances: pd.DataFrame, year: int
) -> pd.DataFrame:
    audit_mask = rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    audit = rows.loc[audit_mask].reset_index(drop=True)
    history = rows.loc[rows["season"].lt(year)].reset_index(drop=True)
    renewal = build_strict_renewal_features(rows, appearances, year)
    anchors = completed_success_anchors(
        history, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    global_prior = float(history[TARGET].mean())
    ids = audit["pitcher_id"].astype(int).to_numpy()
    pairs = [anchors.get(int(value), (0.0, 0.0)) for value in ids]
    anchor_n = np.asarray([item[0] for item in pairs], dtype=np.float64)
    anchor_s = np.asarray([item[1] for item in pairs], dtype=np.float64)
    career_n = pd.to_numeric(
        audit["asof_pitcher_n"], errors="coerce"
    ).to_numpy(np.float64)
    career_rate = pd.to_numeric(
        audit["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    delta_n = np.maximum(np.nan_to_num(career_n - anchor_n, nan=0.0), 0.0)
    delta_s = np.maximum(
        np.nan_to_num(career_n * career_rate - anchor_s, nan=0.0), 0.0
    )
    pitcher_prior = (anchor_s + 400.0 * global_prior) / np.clip(
        anchor_n + 400.0, 1.0, None
    )
    age = renewal["renewal_expected_age"].to_numpy(np.float64)
    age_sd = renewal["renewal_age_sd"].to_numpy(np.float64)
    typical = renewal["renewal_history_median_length"].to_numpy(np.float64)
    completed_n = np.maximum(delta_n - np.nan_to_num(age, nan=0.0), 0.0)

    output = pd.DataFrame(
        {
            "appearance_expected_age": age,
            "appearance_age_sd": age_sd,
            "appearance_season_n": delta_n,
            "appearance_season_successes": delta_s,
            "appearance_pitcher_prior": pitcher_prior,
            "appearance_typical_length": typical,
            "appearance_renewal_covered": renewal["renewal_covered"].to_numpy(
                np.float64
            ),
        }
    )
    estimates: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        success = pd.to_numeric(
            audit[f"asof_pitcher_prev{window}_game_success_rate"], errors="coerce"
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            audit[f"asof_pitcher_prev{window}_game_middle_rate"], errors="coerce"
        ).to_numpy(np.float64)
        minimum, ambiguity, _error = decode_shared_denominator(
            success, middle, MAX_DENOMINATOR[window]
        )
        expected_n = np.clip(typical * float(window), 1.0, MAX_DENOMINATOR[window])
        refined = refine_denominator(
            minimum, success, middle, expected_n, MAX_DENOMINATOR[window]
        )
        usable_n = np.minimum(np.nan_to_num(refined, nan=0.0), completed_n)
        completed_success_est = (
            np.nan_to_num(success, nan=pitcher_prior) * usable_n
            + pitcher_prior * np.maximum(completed_n - usable_n, 0.0)
        )
        current_success_est = delta_s - completed_success_est
        estimates[window] = current_success_est
        expected_success = age * pitcher_prior
        standardized = (current_success_est - expected_success) / np.sqrt(
            np.maximum(age * pitcher_prior * (1.0 - pitcher_prior), 1.0)
        )
        output[f"appearance_prev{window}_refined_n"] = refined
        output[f"appearance_prev{window}_ambiguity"] = ambiguity
        output[f"appearance_prev{window}_success_est"] = current_success_est
        output[f"appearance_prev{window}_excess_z"] = np.clip(standardized, -6.0, 6.0)
        for shrink in (5.0, 15.0, 30.0):
            output[f"appearance_prev{window}_rate_k{int(shrink)}"] = np.clip(
                (current_success_est + shrink * pitcher_prior)
                / np.clip(age + shrink, 1.0, None),
                0.0,
                1.0,
            )
    output["appearance_success_est_mean"] = np.nanmean(
        np.column_stack([estimates[window] for window in WINDOWS]), axis=1
    )
    output["appearance_success_est_spread"] = np.nanstd(
        np.column_stack([estimates[window] for window in WINDOWS]), axis=1
    )
    output["appearance_age_relative_uncertainty"] = age_sd / np.clip(age + 5.0, 5.0, None)
    return output.astype(np.float64)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def audit_metrics(
    rows: pd.DataFrame, features: pd.DataFrame, year: int
) -> dict[str, Any]:
    mask = rows["season"].eq(year) & rows["game_type"].astype(str).eq("R")
    audit = rows.loc[mask].reset_index(drop=True)
    true_age = audit["_appearance_age"].to_numpy(np.float64)
    true_s = audit["_appearance_success_before"].to_numpy(np.float64)
    prior = features["appearance_pitcher_prior"].to_numpy(np.float64)
    true_excess = (true_s - true_age * prior) / np.sqrt(
        np.maximum(true_age * prior * (1.0 - prior), 1.0)
    )
    covered = (
        features["appearance_renewal_covered"].to_numpy(bool)
        & np.isfinite(features["appearance_success_est_mean"].to_numpy())
    )
    age10 = covered & (true_age >= 10.0)
    result: dict[str, Any] = {
        "rows": int(len(audit)),
        "coverage": float(covered.mean()),
        "age10_rows": int(age10.sum()),
        "mean_success_count_correlation_age10": _safe_corr(
            true_s[age10],
            features.loc[age10, "appearance_success_est_mean"].to_numpy(np.float64),
        ),
    }
    for window in WINDOWS:
        estimate = features[f"appearance_prev{window}_success_est"].to_numpy(np.float64)
        excess = features[f"appearance_prev{window}_excess_z"].to_numpy(np.float64)
        result[f"prev{window}_success_count_correlation_age10"] = _safe_corr(
            true_s[age10], estimate[age10]
        )
        result[f"prev{window}_excess_correlation_age10"] = _safe_corr(
            true_excess[age10], excess[age10]
        )
        hot = true_excess[age10] > 0.0
        result[f"prev{window}_hot_auc_age10"] = float(
            roc_auc_score(hot, excess[age10])
        ) if np.unique(hot).size == 2 else float("nan")
    return result


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_anchors_and_renewal_prior": True,
        "audit_labels_used_only_for_reconstruction_metrics": True,
    }


def run(train_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "row_id", "season", "game_type", "inning", "top_bottom", "pitcher_id",
        "asof_pitcher_n", "asof_pitcher_success_rate", TARGET,
        *[
            f"asof_pitcher_prev{window}_game_{rate}_rate"
            for window in WINDOWS for rate in ("success", "middle")
        ],
    ]
    raw = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    rows, appearances = attach_appearance_success(raw)
    metrics: dict[str, Any] = {}
    for year in AUDIT_YEARS:
        features = build_current_appearance_features(rows, appearances, year)
        metrics[str(year)] = audit_metrics(rows, features, year)
        features.to_parquet(
            output_dir / f"current_appearance_features_{year}.parquet", index=False
        )
    eligible = bool(all(
        item["coverage"] >= 0.75
        and item["mean_success_count_correlation_age10"] >= 0.30
        and max(item[f"prev{window}_hot_auc_age10"] for window in WINDOWS) >= 0.60
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
