"""Validate a low-capacity same-regime residual calibrator on Futures rows.

The regular league changed to ABS in 2024, so a 2024-R residual calibrator for
2025 cannot be validated by ordinary year-forward R folds.  Futures provides
the natural analogue because its break occurred in 2023.  This experiment:

1. fits group residual corrections on August-2023 F above the v320 parent;
2. selects one frozen recipe on September/October-2023 F;
3. refits that recipe on all late-2023 F and audits full-2024 F once.

Only if this transport succeeds should the same structure be fitted on 2024 R
for a future 2025 package.  Group keys use only the current row and frozen
parent probability; no evaluation-row aggregation is needed at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V325_WITHIN_REGIME_RESIDUAL_CALIBRATOR_V1"
TARGET = "control_success"
ALPHAS = (50.0, 200.0, 800.0)
WEIGHTS = (0.25, 0.50, 1.00)
PREDICTION_EDGES = (-np.inf, 0.44, 0.48, 0.52, 0.56, np.inf)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "global": ("game_type",),
    "count": ("game_type", "count"),
    "prediction_bin": ("game_type", "prediction_bin"),
    "count_prediction_bin": ("game_type", "count", "prediction_bin"),
    "count_hand": ("game_type", "count", "hand_matchup"),
    "pressure_prediction_bin": ("game_type", "pressure", "prediction_bin"),
}


def add_keys(frame: pd.DataFrame, parent: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["game_type"] = frame["game_type"].astype(str).to_numpy()
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    output["count"] = balls.astype(str) + "-" + strikes.astype(str)
    output["hand_matchup"] = (
        frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    ).to_numpy()
    output["pressure"] = np.select(
        [balls.eq(3), strikes.eq(2)], ["THREE_BALL", "TWO_STRIKE"], default="NEUTRAL"
    )
    output["prediction_bin"] = pd.cut(
        np.asarray(parent, dtype=np.float64), bins=PREDICTION_EDGES, labels=False
    ).astype(str)
    return output


def fit_lookup(
    frame: pd.DataFrame,
    parent: np.ndarray,
    keys: tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    keyed = add_keys(frame, parent)
    keyed["residual"] = frame[TARGET].to_numpy(np.float64) - np.asarray(parent, dtype=np.float64)
    stats = keyed.groupby(list(keys), observed=True)["residual"].agg(n="size", s="sum")
    stats["correction"] = stats["s"] / (stats["n"] + float(alpha))
    return stats


def map_lookup(
    frame: pd.DataFrame,
    parent: np.ndarray,
    keys: tuple[str, ...],
    lookup: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    keyed = add_keys(frame, parent)
    if len(keys) == 1:
        index = pd.Index(keyed[keys[0]], name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(keyed[list(keys)])
    mapped = lookup["correction"].reindex(index)
    active = mapped.notna().to_numpy()
    return mapped.fillna(0.0).to_numpy(np.float64), active


def _candidate(
    parent: np.ndarray, correction: np.ndarray, active: np.ndarray, weight: float
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * correction[active], 0.001, 0.999
    )
    return output


def _metrics(
    frame: pd.DataFrame, parent: np.ndarray, candidate: np.ndarray, active: np.ndarray
) -> dict[str, Any]:
    axes = {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    return paired_metrics(axes, parent, candidate, active)


def run(train_csv: Path, v318_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "pitcher_id", "batter_id", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    late23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    full24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    with np.load(v318_axes, allow_pickle=False) as saved:
        parent23_all = saved["candidate_late_2023"].astype(np.float64)
        parent24_all = saved["candidate_full_2024"].astype(np.float64)
        v320_direction24 = parent24_all - saved["parent_full_2024"].astype(np.float64)
    if len(parent23_all) != len(late23) or len(parent24_all) != len(full24):
        raise ValueError("v318 axis alignment mismatch")

    f23 = late23["game_type"].astype(str).eq("F").to_numpy()
    f24 = full24["game_type"].astype(str).eq("F").to_numpy()
    frame23 = late23.loc[f23].reset_index(drop=True)
    frame24 = full24.loc[f24].reset_index(drop=True)
    parent23 = parent23_all[f23]
    parent24 = parent24_all[f24]
    fit_mask = frame23["game_month"].eq(8).to_numpy()
    valid_mask = frame23["game_month"].ge(9).to_numpy()
    fit_frame = frame23.loc[fit_mask].reset_index(drop=True)
    valid_frame = frame23.loc[valid_mask].reset_index(drop=True)
    fit_parent = parent23[fit_mask]
    valid_parent = parent23[valid_mask]

    trials: list[dict[str, Any]] = []
    for signal, keys in FEATURE_SETS.items():
        for alpha in ALPHAS:
            lookup = fit_lookup(fit_frame, fit_parent, keys, alpha)
            correction, active = map_lookup(valid_frame, valid_parent, keys, lookup)
            for weight in WEIGHTS:
                candidate = _candidate(valid_parent, correction, active, weight)
                result = _metrics(valid_frame, valid_parent, candidate, active)
                trials.append(
                    {
                        "signal": signal,
                        "alpha": alpha,
                        "weight": weight,
                        "gain": result["gain"],
                        "positive_month_fraction": result["positive_month_fraction"],
                        "worst_month_gain": result["worst_month_gain"],
                        "mean_abs_shift_active": result["mean_abs_shift_active"],
                        "active_rows": result["active_rows"],
                        "passes": bool(
                            result["gain"] > 0.0
                            and result["positive_month_fraction"] == 1.0
                            and result["worst_month_gain"] > 0.0
                        ),
                    }
                )
    grid = pd.DataFrame(trials).sort_values(
        ["passes", "worst_month_gain", "gain"], ascending=False
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "within_2023_source_grid.csv", index=False)
    chosen = grid.iloc[0]
    recipe = {
        "signal": str(chosen["signal"]),
        "alpha": float(chosen["alpha"]),
        "weight": float(chosen["weight"]),
    }
    keys = FEATURE_SETS[recipe["signal"]]
    lookup = fit_lookup(frame23, parent23, keys, recipe["alpha"])
    correction24, active24_f = map_lookup(frame24, parent24, keys, lookup)
    candidate24 = _candidate(parent24, correction24, active24_f, recipe["weight"])
    locked = _metrics(frame24, parent24, candidate24, active24_f)

    # Embed the F-only audit candidate back into the complete 2024 axis.
    candidate24_all = parent24_all.copy()
    candidate24_all[f24] = candidate24
    active24_all = np.zeros(len(full24), dtype=bool)
    active24_all[f24] = active24_f
    axes_all = {
        "target": full24[TARGET].to_numpy(np.float64),
        "game_month": full24["game_month"].to_numpy(np.int16),
        "pitcher_id": full24["pitcher_id"].to_numpy(),
        "batter_id": full24["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(full24), dtype=bool),
    }
    whole = paired_metrics(axes_all, parent24_all, candidate24_all, active24_all)
    robustness = _robustness(
        axes_all,
        parent24_all,
        candidate24_all,
        active24_all,
        [parent24_all, candidate24_all],
    )
    increment = candidate24_all - parent24_all
    support = (np.abs(increment) > 0.0) | (np.abs(v320_direction24) > 0.0)
    correlation = (
        float(np.corrcoef(increment[support], v320_direction24[support])[0, 1])
        if support.sum() > 2 and np.std(increment[support]) > 0 and np.std(v320_direction24[support]) > 0
        else 0.0
    )
    source_pass = bool(chosen["passes"])
    locked_pass = bool(
        source_pass
        and locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent24_all,
        candidate_full_2024=candidate24_all,
        direction_full_2024=increment,
        active_full_2024=active24_all,
    )
    lookup.reset_index().to_csv(output_dir / "selected_lookup.csv", index=False)
    summary = {
        "protocol": PROTOCOL,
        "status": "transport_candidate" if locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "analogue": "2023-F post-break -> 2024-F post-break",
        "selected_recipe": recipe,
        "source_grid_size": int(len(grid)),
        "source_passing_recipes": int(grid["passes"].sum()),
        "source_validation": chosen.to_dict(),
        "locked_f_2024": locked,
        "locked_whole_2024": whole,
        "locked_whole_rms_shift": float(np.sqrt(np.mean(np.square(increment)))),
        "locked_robustness": robustness,
        "direction_correlation_with_v320_increment": correlation,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "same_regime_analogue": True,
            "source_fit_august_2023_f_only": True,
            "source_selection_september_october_2023_f_only": True,
            "full_2024_used_only_after_recipe_freeze": True,
            "row_local_runtime_keys": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v318_axes, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
