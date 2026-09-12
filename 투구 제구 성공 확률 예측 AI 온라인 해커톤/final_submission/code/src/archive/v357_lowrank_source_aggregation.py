"""Audit source-season aggregation for the frozen v50 low-rank expert.

v335 averages every completed source-season effect, including an exact zero
when a pitcher did not appear in a source season.  That is conservative but
can over-shrink recently established pitchers.  This audit freezes the v50
matrix recipe, 0.50 dose, and F/R_ANCHOR routes, and changes only how already
fitted source-season effects are aggregated.

Aggregation is selected on full-2022 and late-2023.  Full-2024 is opened once
as locked confirmation, including the exact v345 parent interaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.champion.v50_low_rank_pitcher_context import (
    TARGET,
    fit_source_matrix,
    map_source_matrix,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V357_LOWRANK_SOURCE_AGGREGATION_V1"
SIGNAL = "lowrank_s300_r2"
FROZEN_DOSE = 0.50
SOURCE_YEARS = (2020, 2021, 2022, 2023)
MODES = (
    "recency_h1_all",
    "recency_h2_all",
    "seen_equal",
    "seen_recency_h1",
    "last_seen",
)


def _load_wave0(path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    if not np.array_equal(target, frame[TARGET].to_numpy(np.float64)):
        raise ValueError(f"wave0 target/order mismatch: {path}")
    return target, parent


def aggregate(
    models: dict[int, dict[str, object]],
    frame: pd.DataFrame,
    audit_year: int,
    mode: str,
) -> np.ndarray:
    years = [year for year in sorted(models) if year < audit_year]
    mapped = [map_source_matrix(models[year], frame) for year in years]
    values = np.vstack([item[0][SIGNAL] for item in mapped])
    seen = np.vstack([item[1] for item in mapped]).astype(np.float64)
    age = np.asarray([(audit_year - 1) - year for year in years], dtype=np.float64)
    if mode == "equal_all":
        return values.mean(axis=0)
    if mode in {"recency_h1_all", "recency_h2_all"}:
        half_life = 1.0 if mode == "recency_h1_all" else 2.0
        weight = np.power(0.5, age / half_life)
        return np.average(values, axis=0, weights=weight)
    if mode in {"seen_equal", "seen_recency_h1"}:
        weight = np.ones(len(years), dtype=np.float64)
        if mode == "seen_recency_h1":
            weight = np.power(0.5, age)
        weighted_seen = weight[:, None] * seen
        denominator = weighted_seen.sum(axis=0)
        numerator = (weight[:, None] * values).sum(axis=0)
        return np.divide(
            numerator, denominator,
            out=np.zeros(values.shape[1], dtype=np.float64),
            where=denominator > 0.0,
        )
    if mode == "last_seen":
        output = np.zeros(values.shape[1], dtype=np.float64)
        for index in range(len(years)):
            active = seen[index].astype(bool)
            output[active] = values[index, active]
        return output
    raise ValueError(f"unknown aggregation mode: {mode}")


def apply_replacement(
    parent: np.ndarray,
    equal_direction: np.ndarray,
    alternative_direction: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active]
        + FROZEN_DOSE * (alternative_direction[active] - equal_direction[active]),
        0.001,
        0.999,
    )
    return output


def run(
    train_csv: Path,
    wave0_dir: Path,
    v314_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_team_id", "batter_team_id",
        "pitcher_id", "balls_before", "strikes_before", "batter_hand", TARGET,
    ]
    raw = _add_domain_and_pressure(
        pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    )
    yearly = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in SOURCE_YEARS + (2024,)
    }
    models: dict[int, dict[str, object]] = {}
    for year in SOURCE_YEARS:
        target, parent = _load_wave0(
            wave0_dir / f"wave0_incumbent_validate_{year}.npz", yearly[year]
        )
        models[year] = fit_source_matrix(
            yearly[year], target, parent,
            smoothing_grid=(300.0,), rank_grid=(2,),
        )

    late23 = yearly[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": yearly[2022],
        "late_2023": yearly[2023].loc[late23].reset_index(drop=True),
        "full_2024": yearly[2024],
    }
    equal = {
        "full_2022": aggregate(models, yearly[2022], 2022, "equal_all"),
        "late_2023": aggregate(models, yearly[2023], 2023, "equal_all")[late23],
        "full_2024": aggregate(models, yearly[2024], 2024, "equal_all"),
    }
    with np.load(v314_axes, allow_pickle=False) as saved:
        parity = {
            name: float(np.max(np.abs(equal[name] - saved[f"direction_{name}"])))
            for name in frames
        }
    if max(parity.values()) > 1e-12:
        raise ValueError(f"v50 equal-aggregation parity failed: {parity}")

    with np.load(v335_axes, allow_pickle=False) as saved:
        v335 = {
            name: saved[f"candidate_{name}"].astype(np.float64) for name in frames
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        v345_2024 = saved["candidate_full_2024"].astype(np.float64)
    active = {
        name: frames[name]["domain3"].astype(str).isin(["F", "R_ANCHOR"]).to_numpy()
        for name in frames
    }

    directions: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, Any] = {}
    for mode in MODES:
        directions[mode] = {
            "full_2022": aggregate(models, yearly[2022], 2022, mode),
            "late_2023": aggregate(models, yearly[2023], 2023, mode)[late23],
            "full_2024": aggregate(models, yearly[2024], 2024, mode),
        }
        results[mode] = {}
        for name in frames:
            candidate = apply_replacement(
                v335[name], equal[name], directions[mode][name], active[name]
            )
            results[mode][name] = metrics(
                frames[name], frames[name][TARGET].to_numpy(np.float64),
                v335[name], candidate, active[name],
            )

    eligible = []
    for mode in MODES:
        source = results[mode]
        if (
            source["full_2022"]["gain"] > 0.0
            and source["late_2023"]["gain"] > 0.0
            and source["full_2022"]["positive_month_fraction"] >= 0.75
            and source["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and min(
                source["full_2022"]["worst_month_gain"],
                source["late_2023"]["worst_month_gain"],
            ) > -5.0
        ):
            eligible.append(mode)
    selected_mode = max(
        eligible or list(MODES),
        key=lambda mode: (
            min(results[mode]["full_2022"]["gain"], results[mode]["late_2023"]["gain"]),
            np.mean([results[mode]["full_2022"]["gain"], results[mode]["late_2023"]["gain"]]),
            -MODES.index(mode),
        ),
    )
    locked_candidate = apply_replacement(
        v345_2024, equal["full_2024"],
        directions[selected_mode]["full_2024"], active["full_2024"],
    )
    locked_vs_v345 = metrics(
        frames["full_2024"], frames["full_2024"][TARGET].to_numpy(np.float64),
        v345_2024, locked_candidate, active["full_2024"],
    )
    source_pass = selected_mode in eligible
    locked_pass = bool(
        locked_vs_v345["gain"] >= 1.0
        and locked_vs_v345["positive_month_fraction"] >= 0.75
        and locked_vs_v345["worst_month_gain"] > -5.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=v345_2024,
        candidate_full_2024=locked_candidate,
        equal_direction_full_2024=equal["full_2024"],
        selected_direction_full_2024=directions[selected_mode]["full_2024"],
        active_full_2024=active["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "selected_mode_from_sources": selected_mode,
        "eligible_source_modes": eligible,
        "v314_equal_direction_parity_max_abs": parity,
        "frozen_recipe": {
            "signal": SIGNAL,
            "dose": FROZEN_DOSE,
            "routes": ["F", "R_ANCHOR"],
            "only_changed_quantity": "source-season aggregation",
        },
        "mode_results_vs_v335": results,
        "locked_full_2024_vs_v345": locked_vs_v345,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_source_matrices": True,
            "full_2024_not_used_for_mode_selection": True,
            "v50_matrix_recipe_route_and_dose_frozen": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
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
    parser.add_argument("--wave0-dir", type=Path, required=True)
    parser.add_argument("--v314-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.wave0_dir, args.v314_axes, args.v335_axes,
        args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
