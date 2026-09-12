"""Audit v194's context gate under multiplicity and monthly reselection.

The local dose is frozen at v184's source-selected 0.05.  All 130 predeclared
row-local gates therefore form the Reality Check family; 2024 does not choose
the dose.  A leave-one-origin-month-out procedure independently estimates the
gate-selection rule without evaluating a month on labels used to select it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v194_hierarchical_context_transport import (
    AXES,
    GATE_COLUMNS,
    apply_local,
    gate_library,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V195_CONTEXT_GATE_MULTIPLICITY_AUDIT_V1"
FROZEN_WEIGHT = 0.05


def _gain(target: np.ndarray, base: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    y = np.asarray(target, dtype=np.float64)[selected]
    incumbent = np.asarray(base, dtype=np.float64)[selected]
    trial = np.asarray(candidate, dtype=np.float64)[selected]
    if len(y) == 0 or y.mean() in (0.0, 1.0):
        return float("nan")
    improvement = np.square(incumbent - y) - np.square(trial - y)
    return float(100_000.0 * improvement.mean() / (y.mean() * (1.0 - y.mean())))


def _select_gate(
    candidates: dict[str, dict[str, np.ndarray]],
    axes: dict[str, dict[str, np.ndarray]],
    base: dict[str, np.ndarray],
    exclusions: dict[str, np.ndarray] | None = None,
) -> str:
    """Select a minimax gate using only rows not marked by exclusions."""
    rows = []
    for name, predictions in candidates.items():
        gains = []
        for axis_name in AXES:
            mask = np.asarray(axes[axis_name]["exact_mask"], dtype=bool).copy()
            if exclusions is not None:
                mask &= ~np.asarray(exclusions[axis_name], dtype=bool)
            gains.append(
                _gain(axes[axis_name]["target"], base[axis_name], predictions[axis_name], mask)
            )
        rows.append((name, min(gains), float(np.mean(gains))))
    eligible = [row for row in rows if row[1] > 0.0]
    ranked = eligible if eligible else rows
    return max(ranked, key=lambda row: (row[1], row[2], row[0]))[0]


def run(
    train_csv: Path,
    contract_dir: Path,
    bridge_oof: Path,
    v192_axes: Path,
    v184_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(train_csv, usecols=list(GATE_COLUMNS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    raw = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = raw[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": raw[2022].reset_index(drop=True),
        "late_2023": raw[2023].loc[late23].reset_index(drop=True),
        "full_2024": raw[2024].reset_index(drop=True),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    with np.load(v192_axes, allow_pickle=False) as saved:
        base = {axis: saved[axis].astype(np.float64) for axis in AXES}
    local = {
        "full_2022": np.load(v184_dir / "hierarchical_residual_2022.npy", allow_pickle=False).astype(np.float64),
        "late_2023": np.load(v184_dir / "hierarchical_residual_2023.npy", allow_pickle=False).astype(np.float64)[late23],
        "full_2024": np.load(v184_dir / "hierarchical_residual_2024.npy", allow_pickle=False).astype(np.float64),
    }
    gates = gate_library(frames)
    candidates = {
        name: {
            axis: apply_local(
                base[axis], local[axis], axes[axis]["exact_mask"],
                masks[axis]
                & (np.asarray(axes[axis]["domain3"]).astype(str) == "R_CORE"),
                FROZEN_WEIGHT,
            )
            for axis in AXES
        }
        for name, masks in gates.items()
    }
    selected_name = _select_gate(candidates, axes, base)
    selected = candidates[selected_name]
    final_metrics = {
        axis: metrics(axes[axis], base[axis], selected[axis]) for axis in AXES
    }

    # The no-op is included and all gates use the source-frozen dose.
    family24 = [candidates[name]["full_2024"] for name in sorted(candidates)]
    family24.append(base["full_2024"].copy())
    exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
    robust = _robustness(
        axes["full_2024"], base["full_2024"], selected["full_2024"],
        exact24, family24,
    )

    crossfit = {axis: base[axis].copy() for axis in AXES}
    held_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for held_axis in AXES:
        months = np.asarray(axes[held_axis]["game_month"])
        for month in sorted(np.unique(months).tolist()):
            exclusions = {
                axis: np.zeros(len(base[axis]), dtype=bool) for axis in AXES
            }
            held = months == month
            exclusions[held_axis] = held
            gate = _select_gate(candidates, axes, base, exclusions)
            selected_counts[gate] += 1
            crossfit[held_axis][held] = candidates[gate][held_axis][held]
            exact_held = np.asarray(axes[held_axis]["exact_mask"], dtype=bool) & held
            held_rows.append(
                {
                    "held_axis": held_axis,
                    "held_month": int(month),
                    "selected_gate": gate,
                    "held_gain": _gain(
                        axes[held_axis]["target"], base[held_axis],
                        candidates[gate][held_axis], exact_held,
                    ),
                }
            )
    held_table = pd.DataFrame(held_rows)
    held_table.to_csv(
        output_dir / "leave_one_origin_month_out.csv", index=False, encoding="utf-8-sig"
    )
    crossfit_metrics = {
        axis: metrics(axes[axis], base[axis], crossfit[axis]) for axis in AXES
    }

    # Forward diagnostic: older origins plus March--June 2024 choose a gate;
    # July--October 2024 are evaluated once under that choice.
    early24 = np.asarray(axes["full_2024"]["game_month"]) <= 6
    forward_exclusions = {
        axis: np.zeros(len(base[axis]), dtype=bool) for axis in AXES
    }
    forward_exclusions["full_2024"] = ~early24
    forward_gate = _select_gate(candidates, axes, base, forward_exclusions)
    late24 = exact24 & ~early24
    forward_gain = _gain(
        axes["full_2024"]["target"], base["full_2024"],
        candidates[forward_gate]["full_2024"], late24,
    )

    crossfit_pass = all(
        crossfit_metrics[axis]["gain"] > 0.0
        and crossfit_metrics[axis]["positive_month_fraction"] >= 0.5
        for axis in AXES
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    final_pass = all(final_metrics[axis]["gain"] > 0.0 for axis in AXES)
    passed = bool(final_pass and crossfit_pass and robust_pass and forward_gain > 0.0)
    summary = {
        "protocol": PROTOCOL,
        "status": "multiplicity_pass" if passed else "multiplicity_reject",
        "frozen_weight_from_v184_sources": FROZEN_WEIGHT,
        "gate_count": len(gates),
        "selected_gate": selected_name,
        "final_metrics": final_metrics,
        "robustness_2024": robust,
        "leave_one_origin_month_out": {
            "metrics": crossfit_metrics,
            "positive_held_month_fraction": float((held_table["held_gain"] > 0.0).mean()),
            "worst_held_month_gain": float(held_table["held_gain"].min()),
            "selection_counts": dict(selected_counts),
        },
        "forward_2024": {
            "fit_months": [3, 4, 5, 6],
            "audit_months": [7, 8, 9, 10],
            "selected_gate": forward_gate,
            "audit_gain": forward_gain,
        },
        "final_passed": bool(final_pass),
        "crossfit_passed": bool(crossfit_pass),
        "robust_passed": bool(robust_pass),
        "eligible_for_packaging": passed,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "v184_source_selected_weight_frozen": True,
        "all_predeclared_gates_in_reality_check": True,
        "held_month_excluded_from_crossfit_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v192-axes", type=Path, required=True)
    parser.add_argument("--v184-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.bridge_oof,
        args.v192_axes, args.v184_dir, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
