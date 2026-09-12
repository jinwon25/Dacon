"""Re-audit source-qualified v50 ranks on the untouched R_CORE route.

v335 deploys the frozen rank-2 v50 signal only on F and R_ANCHOR.  The older
v50 source table also contains small-dose rank-4/rank-6 R_CORE recipes that
passed both source origins, but v314 later tested only rank-2 at dose 0.50 on
ALL rows.  This experiment restricts the candidate set to the exact R_CORE
recipes that had already passed v50's source-only consensus gate, re-evaluates
them above v335 on full-2022 and late-2023, and opens full-2024 once above the
actual v345 parent.
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


PROTOCOL = "V358_RCORE_LOWRANK_RANK_EXTENSION_V1"
SOURCE_YEARS = (2020, 2021, 2022, 2023)


def _load_wave0(path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    if not np.array_equal(target, frame[TARGET].to_numpy(np.float64)):
        raise ValueError(f"wave0 target/order mismatch: {path}")
    return target, parent


def _bank(
    models: dict[int, dict[str, object]], frame: pd.DataFrame, audit_year: int
) -> dict[str, np.ndarray]:
    years = [year for year in sorted(models) if year < audit_year]
    mapped = [map_source_matrix(models[year], frame)[0] for year in years]
    return {
        name: np.mean(np.vstack([part[name] for part in mapped]), axis=0)
        for name in sorted(mapped[0])
    }


def _apply(
    parent: np.ndarray, direction: np.ndarray, weight: float, active: np.ndarray
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * direction[active], 0.001, 0.999
    )
    return output


def run(
    train_csv: Path,
    wave0_dir: Path,
    v50_consensus_csv: Path,
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
            smoothing_grid=(300.0, 600.0), rank_grid=(2, 4, 6),
        )

    original = pd.read_csv(v50_consensus_csv)
    qualified = original.loc[
        original["domain"].astype(str).eq("R_CORE")
        & original["passes_consensus_gate"].astype(bool)
    ].copy()
    recipes = sorted({
        (str(row.signal), float(row.weight))
        for row in qualified.itertuples(index=False)
    })
    if not recipes:
        raise ValueError("no source-qualified v50 R_CORE recipe")

    late23 = yearly[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": yearly[2022],
        "late_2023": yearly[2023].loc[late23].reset_index(drop=True),
        "full_2024": yearly[2024],
    }
    banks = {
        "full_2022": _bank(models, yearly[2022], 2022),
        "late_2023": {
            name: values[late23]
            for name, values in _bank(models, yearly[2023], 2023).items()
        },
        "full_2024": _bank(models, yearly[2024], 2024),
    }
    with np.load(v314_axes, allow_pickle=False) as saved:
        parity = {
            name: float(np.max(np.abs(
                banks[name]["lowrank_s300_r2"] - saved[f"direction_{name}"]
            ))) for name in frames
        }
    if max(parity.values()) > 1e-12:
        raise ValueError(f"v50 direction parity failed: {parity}")
    with np.load(v335_axes, allow_pickle=False) as saved:
        v335 = {
            name: saved[f"candidate_{name}"].astype(np.float64) for name in frames
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        v345_2024 = saved["candidate_full_2024"].astype(np.float64)
    active = {
        name: frames[name]["domain3"].astype(str).eq("R_CORE").to_numpy()
        for name in frames
    }

    results: dict[str, Any] = {}
    eligible: list[str] = []
    for signal, weight in recipes:
        key = f"{signal}__w{weight:g}"
        results[key] = {
            "original_v50_source": qualified.loc[
                qualified["signal"].astype(str).eq(signal)
                & np.isclose(qualified["weight"].astype(float), weight)
            ].iloc[0].to_dict()
        }
        for name in frames:
            parent = v345_2024 if name == "full_2024" else v335[name]
            candidate = _apply(parent, banks[name][signal], weight, active[name])
            results[key][name] = metrics(
                frames[name], frames[name][TARGET].to_numpy(np.float64),
                parent, candidate, active[name],
            )
        source = results[key]
        if (
            source["full_2022"]["gain"] > 0.0
            and source["late_2023"]["gain"] > 0.0
            and source["full_2022"]["positive_month_fraction"] >= 0.75
            and source["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and min(
                source["full_2022"]["worst_month_gain"],
                source["late_2023"]["worst_month_gain"],
            ) > -3.0
        ):
            eligible.append(key)

    selected = max(
        eligible or list(results),
        key=lambda key: (
            min(results[key]["full_2022"]["gain"], results[key]["late_2023"]["gain"]),
            np.mean([results[key]["full_2022"]["gain"], results[key]["late_2023"]["gain"]]),
            -list(results).index(key),
        ),
    )
    selected_signal, selected_weight_text = selected.split("__w")
    selected_weight = float(selected_weight_text)
    locked = results[selected]["full_2024"]
    source_pass = selected in eligible
    locked_pass = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.75
        and locked["worst_month_gain"] > -5.0
    )
    candidate24 = _apply(
        v345_2024, banks["full_2024"][selected_signal],
        selected_weight, active["full_2024"],
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=v345_2024,
        candidate_full_2024=candidate24,
        direction_full_2024=banks["full_2024"][selected_signal],
        active_full_2024=active["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "original_source_qualified_recipe_count": len(recipes),
        "current_parent_source_eligible": eligible,
        "selected_recipe_from_sources": {
            "key": selected,
            "signal": selected_signal,
            "weight": selected_weight,
            "route": "R_CORE",
        },
        "v314_rank2_direction_parity_max_abs": parity,
        "results": results,
        "locked_full_2024_vs_v345": locked,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "candidate_recipes_prequalified_by_v50_sources": True,
            "current_parent_selection_uses_full_2022_and_late_2023_only": True,
            "full_2024_opened_once_after_selection": True,
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
    parser.add_argument("--v50-consensus-csv", type=Path, required=True)
    parser.add_argument("--v314-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.wave0_dir, args.v50_consensus_csv, args.v314_axes,
        args.v335_axes, args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
