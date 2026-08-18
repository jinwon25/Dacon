"""Screen hierarchical current-season posteriors above v21.

The supplied cumulative ASOF counters mix a player's earlier seasons with the
current season.  This experiment separates those counts, uses a recency-weighted
preseason empirical-Bayes player prior, and then updates that prior with only
the evidence already available before the current row.  Every feature is thus
row-local at inference and season-forward in validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v20_residual_overlay_screen import _bss
from src.v22_domain_calibration_screen import AXES


HALF_LIVES = (0.5, 1.0, 2.0, 4.0)
HISTORY_STRENGTHS = (20.0, 80.0, 200.0)
SEASON_STRENGTHS = (20.0, 40.0, 80.0, 160.0)
PITCHER_FRACTIONS = (0.50, 0.75, 0.90, 1.00)
BLEND_WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)


def _rounded_count(n: np.ndarray, rate: np.ndarray) -> np.ndarray:
    output = np.zeros(len(n), dtype=np.float64)
    valid = (n > 0.0) & np.isfinite(rate)
    output[valid] = np.rint(n[valid] * rate[valid])
    return output


def _audit_state(
    train: pd.DataFrame, audit_year: int, entity: str, n_column: str, rate_column: str
) -> dict[str, np.ndarray]:
    history = train.loc[train["season"].lt(audit_year)]
    audit = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
    history_stats = history.groupby(entity, observed=True)["control_success"].agg(
        history_n="size", history_s="sum"
    )
    history_n = audit[entity].map(history_stats["history_n"]).fillna(0.0).to_numpy(np.float64)
    history_s = audit[entity].map(history_stats["history_s"]).fillna(0.0).to_numpy(np.float64)
    cumulative_n = pd.to_numeric(audit[n_column], errors="coerce").fillna(0.0).to_numpy(np.float64)
    cumulative_rate = pd.to_numeric(audit[rate_column], errors="coerce").to_numpy(np.float64)
    cumulative_s = _rounded_count(cumulative_n, cumulative_rate)
    season_n = np.maximum(cumulative_n - history_n, 0.0)
    season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
    return {
        "ids": audit[entity].to_numpy(),
        "season_n": season_n,
        "season_s": season_s,
    }


def _prior_vectors(
    train: pd.DataFrame,
    audit_year: int,
    entity: str,
    audit_ids: np.ndarray,
) -> dict[tuple[float, float], np.ndarray]:
    history = train.loc[train["season"].lt(audit_year), ["season", entity, "control_success"]].copy()
    output: dict[tuple[float, float], np.ndarray] = {}
    for half_life in HALF_LIVES:
        row_weight = np.exp2(
            -(audit_year - 1.0 - history["season"].to_numpy(np.float64)) / half_life
        )
        weighted_success = row_weight * history["control_success"].to_numpy(np.float64)
        stats = pd.DataFrame(
            {
                entity: history[entity].to_numpy(),
                "weighted_n": row_weight,
                "weighted_s": weighted_success,
            }
        ).groupby(entity, observed=True)[["weighted_n", "weighted_s"]].sum()
        global_rate = float(weighted_success.sum() / row_weight.sum())
        n = pd.Series(audit_ids).map(stats["weighted_n"]).fillna(0.0).to_numpy(np.float64)
        s = pd.Series(audit_ids).map(stats["weighted_s"]).fillna(0.0).to_numpy(np.float64)
        for strength in HISTORY_STRENGTHS:
            output[(half_life, strength)] = (s + strength * global_rate) / (n + strength)
    return output


def _posterior(state: dict[str, np.ndarray], prior: np.ndarray, strength: float) -> np.ndarray:
    return (state["season_s"] + strength * prior) / (state["season_n"] + strength)


def _axis_mask(axis: str, year: int, month: np.ndarray) -> np.ndarray:
    if axis == "y2023_to_y2024":
        if year != 2024:
            raise ValueError(axis)
        return np.ones(len(month), dtype=bool)
    if axis == "y2023_early_to_late":
        if year != 2023:
            raise ValueError(axis)
        return month >= 8
    if axis == "y2024_early_to_late":
        if year != 2024:
            raise ValueError(axis)
        return month >= 8
    raise ValueError(axis)


def run(project: Path, champion_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season",
        "game_month",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "control_success",
    ]
    train = pd.read_csv(project / "data" / "train.csv", usecols=columns, low_memory=False)
    years: dict[int, dict[str, object]] = {}
    for year in (2023, 2024):
        audit = train.loc[train["season"].eq(year)].reset_index(drop=True)
        pitcher_state = _audit_state(
            train, year, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
        )
        batter_state = _audit_state(
            train, year, "batter_id", "asof_batter_n", "asof_batter_success_rate"
        )
        years[year] = {
            "target": audit["control_success"].to_numpy(np.float64),
            "month": audit["game_month"].to_numpy(np.int16),
            "pitcher_state": pitcher_state,
            "batter_state": batter_state,
            "pitcher_prior": _prior_vectors(train, year, "pitcher_id", pitcher_state["ids"]),
            "batter_prior": _prior_vectors(train, year, "batter_id", batter_state["ids"]),
        }

    folds = {}
    for axis in AXES:
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            folds[axis] = {
                "target": saved["target"].astype(np.float64),
                "v21": saved["v21"].astype(np.float64),
                "month": saved["game_month"].astype(np.int16),
                "domain": saved["domain3"].astype(str),
            }

    rows = []
    signal_count = 0
    for half_life, history_strength, season_strength, pitcher_fraction in itertools.product(
        HALF_LIVES, HISTORY_STRENGTHS, SEASON_STRENGTHS, PITCHER_FRACTIONS
    ):
        key = (half_life, history_strength)
        signals = {}
        for year in (2023, 2024):
            values = years[year]
            pitcher = _posterior(
                values["pitcher_state"], values["pitcher_prior"][key], season_strength
            )
            batter = _posterior(
                values["batter_state"], values["batter_prior"][key], season_strength
            )
            signals[year] = pitcher_fraction * pitcher + (1.0 - pitcher_fraction) * batter
        signal_count += 1
        for weight in BLEND_WEIGHTS:
            gains = {}
            month_fractions = {}
            minimum_domains = {}
            for axis in AXES:
                year = 2023 if axis == "y2023_early_to_late" else 2024
                mask = _axis_mask(axis, year, years[year]["month"])
                fold = folds[axis]
                if not np.array_equal(fold["target"], years[year]["target"][mask]):
                    raise ValueError(f"target order mismatch for {axis}")
                signal = signals[year][mask]
                candidate = np.clip(
                    fold["v21"] + weight * (signal - fold["v21"]), 0.001, 0.999
                )
                gains[axis] = _bss(fold["target"], candidate) - _bss(
                    fold["target"], fold["v21"]
                )
                month_gain = []
                for month in sorted(np.unique(fold["month"])):
                    selected = fold["month"] == month
                    month_gain.append(
                        _bss(fold["target"][selected], candidate[selected])
                        - _bss(fold["target"][selected], fold["v21"][selected])
                    )
                domain_gain = []
                for domain in ("R_CORE", "R_ANCHOR", "F"):
                    selected = fold["domain"] == domain
                    domain_gain.append(
                        _bss(fold["target"][selected], candidate[selected])
                        - _bss(fold["target"][selected], fold["v21"][selected])
                    )
                month_fractions[axis] = float(np.mean(np.asarray(month_gain) > 0.0))
                minimum_domains[axis] = float(min(domain_gain))
            rows.append(
                {
                    "half_life": half_life,
                    "history_strength": history_strength,
                    "season_strength": season_strength,
                    "pitcher_fraction": pitcher_fraction,
                    "blend_weight": weight,
                    **gains,
                    "min_gain": min(gains.values()),
                    "mean_gain": float(np.mean(list(gains.values()))),
                    **{f"{axis}__positive_month_fraction": value for axis, value in month_fractions.items()},
                    **{f"{axis}__minimum_domain_gain": value for axis, value in minimum_domains.items()},
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        ["min_gain", "mean_gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    stable = metrics.loc[
        (metrics["min_gain"] > 0.0)
        & (metrics["y2023_to_y2024__positive_month_fraction"] >= 0.75)
        & (metrics["y2024_early_to_late__positive_month_fraction"] >= 1.0)
        & (metrics["y2023_to_y2024__minimum_domain_gain"] > 0.0)
        & (metrics["y2024_early_to_late__minimum_domain_gain"] > 0.0)
    ]
    summary = {
        "protocol": "V22_HIERARCHICAL_CURRENT_SEASON_POSTERIOR_V1",
        "signal_count": signal_count,
        "candidate_count": int(len(metrics)),
        "strictly_positive": int((metrics["min_gain"] > 0.0).sum()),
        "stability_gate_passed": int(len(stable)),
        "best": metrics.head(40).to_dict("records"),
        "best_stable": stable.head(40).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--champion-dir", type=Path, default=Path("artifacts/champion_oof_20260817_01")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_hierarchical_posterior_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()
