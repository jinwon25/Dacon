"""Apply the established April--September stability gate to v260 experts.

The calendar is inherited from the earlier v70 TrackMan calendar experiment,
not selected from v260's locked-month results.  March and October retain the
exact v244 parent; the frozen v260 command+batter expert is used only during
the main regular-season window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v253_fixed_route_command_dispersion_audit import (
    AXES,
    SOURCE_AXES,
    _load_axes,
)


PROTOCOL = "V261_REGULAR_CALENDAR_EXPERT_GATE_V1"
FIRST_ACTIVE_MONTH = 4
LAST_ACTIVE_MONTH = 9


def apply_calendar_gate(
    baseline: np.ndarray,
    expert: np.ndarray,
    active: np.ndarray,
    month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    month = np.asarray(month)
    if not (baseline.shape == expert.shape == active.shape == month.shape):
        raise ValueError("calendar gate arrays have different shapes")
    calendar = (month >= FIRST_ACTIVE_MONTH) & (month <= LAST_ACTIVE_MONTH)
    selected = active & calendar
    candidate = baseline.copy()
    candidate[selected] = expert[selected]
    return candidate, selected


def run(
    train_csv: Path,
    v260_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(v260_axes, allow_pickle=False) as saved:
        baselines = {
            axis: saved[f"parent_{axis}"].astype(np.float64) for axis in AXES
        }
        experts = {
            axis: saved[f"candidate_{axis}"].astype(np.float64) for axis in AXES
        }
        original_active = {
            axis: saved[f"active_{axis}"].astype(bool) for axis in AXES
        }
    candidates: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    for axis in AXES:
        month = axis_frames[axis]["game_month"].to_numpy()
        candidates[axis], selected[axis] = apply_calendar_gate(
            baselines[axis], experts[axis], original_active[axis], month
        )
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], selected[axis]
        )
    source_pass = all(results[axis]["gain"] > 0.0 for axis in SOURCE_AXES)
    locked_pass = results["full_2024"]["gain"] > 0.0
    robustness = _robustness(
        axes["full_2024"],
        baselines["full_2024"],
        candidates["full_2024"],
        selected["full_2024"],
        [
            candidates["full_2024"],
            experts["full_2024"],
            baselines["full_2024"].copy(),
        ],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.05
    )
    np.savez_compressed(
        output_dir / "candidate_axes.npz",
        **{f"parent_{axis}": baselines[axis] for axis in AXES},
        **{f"candidate_{axis}": candidates[axis] for axis in AXES},
        **{f"active_{axis}": selected[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "local_gate_pass"
            if source_pass and locked_pass and robust_pass
            else "source_reject"
            if not source_pass
            else "locked_or_robust_reject"
        ),
        "calendar_window": [FIRST_ACTIVE_MONTH, LAST_ACTIVE_MONTH],
        "results": results,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(source_pass and locked_pass and robust_pass),
        "eligible_for_packaging": False,
        "restrictions": {
            "calendar_inherited_from_v70_not_selected_from_v260": True,
            "v260_command_batter_formula_frozen": True,
            "march_and_october_exact_v244_parent": True,
            "full_2024_v260_months_observed_before_this_audit": True,
            "exploratory_not_confirmatory": True,
            "test_csv_read": False,
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
    parser.add_argument("--v260-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v260_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
