"""Rebase the frozen v50 low-rank signal above v290 without retuning it."""

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
    build_audit_bank,
    fit_source_matrix,
)


PROTOCOL = "V314_FROZEN_V50_LOWRANK_REBASE_V290_V1"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
SIGNAL = "lowrank_s300_r2"
FROZEN_DOSE = 0.50


def run(
    train_csv: Path,
    wave0_oof_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "pitcher_id", "balls_before", "strikes_before",
        "batter_hand", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    yearly = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in SOURCE_YEARS + (2024,)
    }
    models = {}
    for year in SOURCE_YEARS:
        with np.load(
            wave0_oof_dir / f"wave0_incumbent_validate_{year}.npz",
            allow_pickle=False,
        ) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        expected = yearly[year][TARGET].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"wave0 target/order mismatch for {year}")
        models[year] = fit_source_matrix(
            yearly[year], target, parent,
            smoothing_grid=(300.0,), rank_grid=(2,),
        )

    banks = {
        year: build_audit_bank(models, yearly[year], year)[0][SIGNAL]
        for year in (2022, 2023, 2024)
    }
    late23 = yearly[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": yearly[2022],
        "late_2023": yearly[2023].loc[late23].reset_index(drop=True),
        "full_2024": yearly[2024],
    }
    directions = {
        "full_2022": banks[2022],
        "late_2023": banks[2023][late23],
        "full_2024": banks[2024],
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parents.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    active = {name: np.ones(len(frame), dtype=bool) for name, frame in frames.items()}
    candidates = {
        name: np.clip(parents[name] + FROZEN_DOSE * directions[name], 0.001, 0.999)
        for name in frames
    }
    results = {
        name: metrics(
            frame,
            frame[TARGET].to_numpy(np.float64),
            parents[name], candidates[name], active[name],
        )
        for name, frame in frames.items()
    }
    source_pass = bool(
        results["full_2022"]["gain"] > 0.0
        and results["late_2023"]["gain"] > 0.0
        and results["full_2022"]["positive_month_fraction"] >= 0.75
        and results["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
    )
    locked_pass = bool(
        results["full_2024"]["gain"] > 0.0
        and results["full_2024"]["positive_month_fraction"] >= 0.625
        and results["full_2024"]["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "frozen_recipe": {"signal": SIGNAL, "route": "ALL", "dose": FROZEN_DOSE},
        "metrics": results,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "v50_recipe_retuned": False,
            "v290_parent_preserved_before_addition": True,
            "test_data_used": False,
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
    parser.add_argument("--wave0-oof-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.wave0_oof_dir, args.v285_axes, args.v290_axes, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
