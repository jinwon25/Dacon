"""Audit whether the frozen v320 effect size supports a 1185 claim.

This is a diagnostic, not a recipe selector.  It reports the prediction-space
movement required for the remaining Public gap and performs leave-one-month-
out fits of the two v320 directions only to measure temporal instability.  The
post-hoc fits are explicitly prohibited from changing the deployed weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.archive.v298_recent_rate_denominator_signal import bss


PROTOCOL = "V327_TARGET1185_FEASIBILITY_AUDIT_V1"
TARGET = "control_success"


def required_rms(gain: float, target_rate: float) -> float:
    denominator = float(target_rate * (1.0 - target_rate))
    return float(np.sqrt(float(gain) * denominator / 100000.0))


def run(
    train_csv: Path,
    v318_axes: Path,
    transfer_summary: Path,
    output_dir: Path,
    incumbent_public: float,
    target_public: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "game_type", TARGET],
        low_memory=False,
    )
    frame = frame.loc[frame["season"].eq(2024)].reset_index(drop=True)
    with np.load(v318_axes, allow_pickle=False) as saved:
        parent = saved["parent_full_2024"].astype(np.float64)
        candidate = saved["candidate_full_2024"].astype(np.float64)
        direct = saved["direct_direction_full_2024"].astype(np.float64)
        lowrank = saved["lowrank_direction_full_2024"].astype(np.float64)
    if len(frame) != len(parent):
        raise ValueError("full-2024 alignment mismatch")
    target = frame[TARGET].to_numpy(np.float64)
    target_rate = float(target.mean())
    gap = float(target_public - incumbent_public)
    rms_needed = required_rms(gap, target_rate)
    increment = candidate - parent
    rms_actual = float(np.sqrt(np.mean(np.square(increment))))

    f_active = frame["game_type"].astype(str).eq("F").to_numpy()
    matrix = np.column_stack((direct, lowrank))
    month_rows: list[dict[str, Any]] = []
    for month in sorted(int(value) for value in frame["game_month"].unique()):
        held = frame["game_month"].eq(month).to_numpy() & f_active
        fit = ~frame["game_month"].eq(month).to_numpy() & f_active
        fitted = lsq_linear(
            matrix[fit], target[fit] - parent[fit], bounds=(0.0, 2.0), lsmr_tol="auto"
        )
        held_candidate = parent[held] + matrix[held] @ fitted.x
        month_rows.append(
            {
                "held_month": month,
                "fit_direct_scale": float(fitted.x[0]),
                "fit_lowrank_scale": float(fitted.x[1]),
                "held_rows": int(held.sum()),
                "held_gain": bss(target[held], held_candidate) - bss(target[held], parent[held]),
            }
        )
    pd.DataFrame(month_rows).to_csv(output_dir / "leave_one_month_out.csv", index=False)

    transfer = json.loads(transfer_summary.read_text(encoding="utf-8"))
    median_ratio = float(transfer["median_ratio"])
    total_ratio = float(transfer["total_ratio"])
    scale_diagnostic = {
        "local_gain_needed_at_historical_median_ratio": gap / median_ratio,
        "local_gain_needed_at_historical_total_ratio": gap / total_ratio,
        "warning": "effect-size scenarios only; local-to-Public magnitude forecasting is invalid",
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "not_supported_for_1185_claim",
        "public": {
            "incumbent": incumbent_public,
            "target": target_public,
            "gap": gap,
        },
        "full_2024": {
            "target_rate": target_rate,
            "required_whole_row_rms_for_gap_if_perfectly_aligned": rms_needed,
            "v320_whole_row_rms": rms_actual,
            "rms_fraction_of_requirement": rms_actual / rms_needed,
            "v320_gain": bss(target, candidate) - bss(target, parent),
            "positive_held_month_fraction": float(np.mean([row["held_gain"] > 0 for row in month_rows])),
            "worst_held_month_gain": float(min(row["held_gain"] for row in month_rows)),
            "leave_one_month_out": month_rows,
        },
        "historical_transfer_scale_diagnostic": scale_diagnostic,
        "decision": {
            "v320_can_be_called_1180_certain": False,
            "v320_can_be_called_1185_candidate": False,
            "v320_remains_best_exploratory_submission": True,
            "weights_changed_from_posthoc_audit": False,
        },
        "restrictions": {
            "diagnostic_only": True,
            "leave_one_month_out_weights_not_deployable": True,
            "test_data_used": False,
            "public_score_used_to_change_recipe": False,
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
    parser.add_argument("--transfer-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incumbent-public", type=float, default=1176.757071668)
    parser.add_argument("--target-public", type=float, default=1185.0)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv,
        args.v318_axes,
        args.transfer_summary,
        args.output_dir,
        args.incumbent_public,
        args.target_public,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
