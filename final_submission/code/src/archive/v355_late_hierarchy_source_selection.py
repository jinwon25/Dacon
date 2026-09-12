"""Select a mature-season hierarchy on 2022/late-2023, then audit 2024.

All 54 raw forecasts are the preregistered v39 hierarchy bank.  Route,
August start, and 0.20 dose are fixed by v354.  Only the raw prior variant is
ranked on full-2022 and late-2023; full-2024 remains outside that ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics
from src.archive.v354_late_hierarchy_rebase_v345 import (
    blend_toward_hierarchy,
    late_anchor_mask,
)
from src.archive.v39_hierarchical_season_forecast import forecast_bank
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V355_LATE_HIERARCHY_SOURCE_SELECTION_V1"
TARGET = "control_success"


def evaluate(frame: pd.DataFrame, parent: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate, active = blend_toward_hierarchy(frame, parent, raw)
    metrics = axis_metrics(frame, parent, candidate, active)
    metrics["full_row_rms_shift"] = float(
        np.sqrt(np.mean(np.square(candidate - np.asarray(parent, dtype=np.float64))))
    )
    return candidate, active, metrics


def run(
    train_csv: Path,
    v335_axes: Path,
    v345_axes: Path,
    v352_axes: Path,
    v354_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(train_csv, low_memory=False))
    full22 = train.loc[train["season"].eq(2022)].reset_index(drop=True)
    full23 = train.loc[train["season"].eq(2023)].reset_index(drop=True)
    late23_mask = full23["game_month"].ge(8).to_numpy()
    late23 = full23.loc[late23_mask].reset_index(drop=True)
    full24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    banks = {year: forecast_bank(train, year) for year in (2022, 2023)}
    with np.load(v335_axes, allow_pickle=False) as saved:
        parent22 = saved["candidate_full_2022"].astype(np.float64)
        parent23 = saved["candidate_late_2023"].astype(np.float64)

    rows: list[dict[str, Any]] = []
    for signal in sorted(banks[2022]):
        _candidate22, _active22, metrics22 = evaluate(
            full22, parent22, banks[2022][signal]
        )
        _candidate23, _active23, metrics23 = evaluate(
            late23, parent23, banks[2023][signal][late23_mask]
        )
        rows.append(
            {
                "signal": signal,
                "source_min_gain": min(metrics22["gain"], metrics23["gain"]),
                "source_mean_gain": float(np.mean([metrics22["gain"], metrics23["gain"]])),
                "source_worst_month_gain": min(
                    metrics22["worst_month_gain"], metrics23["worst_month_gain"]
                ),
                "all_source_months_positive": bool(
                    metrics22["positive_month_fraction"] == 1.0
                    and metrics23["positive_month_fraction"] == 1.0
                ),
                "metrics_full_2022": metrics22,
                "metrics_late_2023": metrics23,
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row["all_source_months_positive"]),
            float(row["source_min_gain"]),
            float(row["source_worst_month_gain"]),
            float(row["source_mean_gain"]),
        ),
        reverse=True,
    )
    ranking = pd.DataFrame(
        [
            {key: value for key, value in row.items() if not key.startswith("metrics_")}
            for row in rows
        ]
    )
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = rows[0]
    signal = str(selected["signal"])

    raw24 = forecast_bank(train, 2024)[signal]
    with np.load(v345_axes, allow_pickle=False) as saved:
        v335_2024 = saved["parent_full_2024"].astype(np.float64)
        parent24 = saved["candidate_full_2024"].astype(np.float64)
    candidate24, active24, metrics24 = evaluate(full24, parent24, raw24)
    axes24 = {
        "target": full24[TARGET].to_numpy(np.float64),
        "game_month": full24["game_month"].to_numpy(np.int16),
        "pitcher_id": full24["pitcher_id"].to_numpy(),
        "batter_id": full24["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(full24), dtype=bool),
    }
    robustness = _robustness(
        axes24, parent24, candidate24, active24, [parent24, candidate24]
    )
    with np.load(v352_axes, allow_pickle=False) as saved:
        pfd_increment = saved["increment_late_2024"].astype(np.float64)
        pfd_active = saved["active_late_2024"].astype(bool)
    late24 = full24["game_month"].ge(8).to_numpy()
    hierarchy_increment = candidate24 - parent24
    overlap = active24[late24] & pfd_active
    correlation_pfd = float(
        np.corrcoef(hierarchy_increment[late24][overlap], pfd_increment[overlap])[0, 1]
    )
    v354 = json.loads(v354_summary.read_text(encoding="utf-8"))
    v354_gain = float(v354["metrics"]["full_2024"]["gain"])
    gain_over_v354 = float(metrics24["gain"] - v354_gain)
    overlap_v345 = int(
        np.count_nonzero(active24 & (np.abs(parent24 - v335_2024) > 1e-15))
    )
    passed = bool(
        selected["all_source_months_positive"]
        and selected["source_min_gain"] > 0.0
        and selected["source_worst_month_gain"] > 0.0
        and metrics24["gain"] > v354_gain
        and metrics24["positive_month_fraction"] == 1.0
        and metrics24["worst_month_gain"] > 0.0
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and abs(correlation_pfd) <= 0.30
        and overlap_v345 == 0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parent24,
        candidate_full_2024=candidate24,
        hierarchy_full_2024=raw24,
        hierarchy_increment_full_2024=hierarchy_increment,
        active_full_2024=active24,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "reject",
        "selection": {
            "candidate_count": len(rows),
            "signal": signal,
            "source_min_gain": selected["source_min_gain"],
            "source_mean_gain": selected["source_mean_gain"],
            "source_worst_month_gain": selected["source_worst_month_gain"],
            "metrics_full_2022": selected["metrics_full_2022"],
            "metrics_late_2023": selected["metrics_late_2023"],
        },
        "locked_full_2024": metrics24,
        "gain_over_v354": gain_over_v354,
        "locked_robustness": robustness,
        "diversity": {
            "correlation_with_v352_pfd_on_overlap": correlation_pfd,
            "overlap_rows_with_post_v335_v345_increment": overlap_v345,
        },
        "recipe": {
            "signal": signal,
            "domain": "R_ANCHOR",
            "start_month": 8,
            "weight_toward_hierarchy": 0.20,
        },
        "candidate_gate_passed": passed,
        "selection_warning": (
            "54 highly correlated preregistered v39 hierarchy variants were "
            "ranked on 2022 and late-2023 only. Full-2024 was excluded from "
            "ranking but this family has historical development exposure."
        ),
        "restrictions": {
            "official_train_only": True,
            "source_only_variant_selection": True,
            "full_2024_excluded_from_ranking": True,
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
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--v352-axes", type=Path, required=True)
    parser.add_argument("--v354-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v335_axes, args.v345_axes, args.v352_axes, args.v354_summary, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
