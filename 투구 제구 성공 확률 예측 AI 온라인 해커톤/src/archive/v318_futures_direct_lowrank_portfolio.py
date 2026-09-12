"""Audit the fixed 20%-direct-F plus frozen v50 low-rank portfolio above v290."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V318_FUTURES_DIRECT_LOWRANK_PORTFOLIO_V1"
EXTRA_DIRECT_WEIGHT = 0.10
LOWRANK_WEIGHT = 0.50


def run(
    train_csv: Path,
    v285_axes: Path,
    v290_axes: Path,
    v314_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season", "game_month", "game_type", "pitcher_id", "batter_id",
            "control_success",
        ],
        low_memory=False,
    )
    frames = {
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        exact = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in frames
        }
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in frames
        }
    with np.load(v314_axes, allow_pickle=False) as saved:
        lowrank = {
            name: LOWRANK_WEIGHT * saved[f"direction_{name}"].astype(np.float64)
            for name in frames
        }
    direct = {name: parent[name] - exact[name] for name in frames}
    active = {
        name: frames[name]["game_type"].astype(str).eq("F").to_numpy()
        for name in frames
    }
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for name in frames:
        candidates[name] = {}
        for key, direction in {
            "extra_direct": direct[name],
            "lowrank": lowrank[name],
            "portfolio": direct[name] + lowrank[name],
        }.items():
            candidate = parent[name].copy()
            candidate[active[name]] = np.clip(
                parent[name][active[name]] + direction[active[name]], 0.001, 0.999
            )
            candidates[name][key] = candidate

    axes = {
        name: {
            "target": frame["control_success"].to_numpy(np.float64),
            "game_month": frame["game_month"].to_numpy(np.int16),
            "pitcher_id": frame["pitcher_id"].to_numpy(),
            "batter_id": frame["batter_id"].to_numpy(),
            "exact_mask": np.ones(len(frame), dtype=bool),
        }
        for name, frame in frames.items()
    }
    metrics = {
        name: {
            key: paired_metrics(axes[name], parent[name], candidate, active[name])
            for key, candidate in candidates[name].items()
        }
        for name in frames
    }
    family = [
        parent["full_2024"],
        candidates["full_2024"]["extra_direct"],
        candidates["full_2024"]["lowrank"],
        candidates["full_2024"]["portfolio"],
    ]
    robustness = _robustness(
        axes["full_2024"],
        parent["full_2024"],
        candidates["full_2024"]["portfolio"],
        active["full_2024"],
        family,
    )
    source_pass = bool(
        metrics["late_2023"]["portfolio"]["gain"] > 0.0
        and metrics["late_2023"]["portfolio"]["positive_month_fraction"] == 1.0
    )
    point_pass = bool(
        metrics["full_2024"]["portfolio"]["gain"] >= 3.0
        and metrics["full_2024"]["portfolio"]["positive_month_fraction"] >= 0.625
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    candidate24 = candidates["full_2024"]["portfolio"]
    rms = float(np.sqrt(np.mean(np.square(candidate24 - parent["full_2024"]))))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parent[name] for name in frames},
        **{f"candidate_{name}": candidates[name]["portfolio"] for name in frames},
        **{f"active_{name}": active[name] for name in frames},
        **{f"direct_direction_{name}": direct[name] for name in frames},
        **{f"lowrank_direction_{name}": lowrank[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "package_candidate" if source_pass and point_pass else "reject",
        "fixed_recipe": {
            "v290_direct_f_weight": 0.10,
            "extra_direct_f_weight": EXTRA_DIRECT_WEIGHT,
            "resulting_direct_f_weight": 0.20,
            "v50_lowrank_signal": "s300_r2",
            "v50_lowrank_weight": LOWRANK_WEIGHT,
            "route": "F",
        },
        "metrics": metrics,
        "locked_full_row_rms_shift": rms,
        "locked_robustness": robustness,
        "source_gate_passed": source_pass,
        "locked_point_gate_passed": point_pass,
        "locked_robust_gate_passed": robust_pass,
        "eligible_for_exploratory_packaging": bool(source_pass and point_pass),
        "restrictions": {
            "official_train_only": True,
            "direct_20pct_supported_by_pre_v290_local_curve": True,
            "v50_recipe_frozen_from_pre_v290_two_origin_selection": True,
            "full_2024_used_for_recipe_selection": False,
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
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--v314-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v290_axes, args.v314_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
