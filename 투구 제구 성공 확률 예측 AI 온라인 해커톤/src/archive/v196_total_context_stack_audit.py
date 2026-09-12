"""Audit the total v192 plus v195 context stack against the exact JY parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v194_hierarchical_context_transport import (
    AXES,
    GATE_COLUMNS,
    apply_local,
    gate_library,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V196_TOTAL_CONTEXT_STACK_AUDIT_V1"


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v192_axes: Path,
    v184_dir: Path,
    v195_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v195 = json.loads(v195_summary_path.read_text(encoding="utf-8"))
    if v195.get("protocol") != "V195_CONTEXT_GATE_MULTIPLICITY_AUDIT_V1":
        raise ValueError("expected v195 context evidence")
    selected_gate = str(v195["selected_gate"])
    weight = float(v195["frozen_weight_from_v184_sources"])

    _context, jy_frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, jy_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )

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
    with np.load(v192_axes, allow_pickle=False) as saved:
        base = {axis: saved[axis].astype(np.float64) for axis in AXES}
    local = {
        "full_2022": np.load(v184_dir / "hierarchical_residual_2022.npy", allow_pickle=False).astype(np.float64),
        "late_2023": np.load(v184_dir / "hierarchical_residual_2023.npy", allow_pickle=False).astype(np.float64)[late23],
        "full_2024": np.load(v184_dir / "hierarchical_residual_2024.npy", allow_pickle=False).astype(np.float64),
    }
    gates = gate_library(frames)
    if selected_gate not in gates:
        raise ValueError(f"v195 selected gate is unavailable: {selected_gate}")

    family: dict[str, dict[str, np.ndarray]] = {}
    for name, masks in gates.items():
        family[name] = {}
        for axis in AXES:
            rcore_gate = masks[axis] & (
                np.asarray(axes[axis]["domain3"]).astype(str) == "R_CORE"
            )
            family[name][axis] = apply_local(
                base[axis], local[axis], axes[axis]["exact_mask"],
                rcore_gate, weight,
            )
    selected = family[selected_gate]
    final_metrics = {
        axis: metrics(axes[axis], parents[axis], selected[axis]) for axis in AXES
    }
    base_metrics = {
        axis: metrics(axes[axis], parents[axis], base[axis]) for axis in AXES
    }

    ordered_names = sorted(family)
    family24 = [family[name]["full_2024"] for name in ordered_names]
    family24.extend([base["full_2024"], parents["full_2024"]])
    exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
    robust = _robustness(
        axes["full_2024"], parents["full_2024"], selected["full_2024"],
        exact24, family24,
    )
    family_gains = {
        name: metrics(
            axes["full_2024"], parents["full_2024"],
            family[name]["full_2024"],
        )["gain"]
        for name in ordered_names
    }
    ranked_names = sorted(family_gains, key=family_gains.get, reverse=True)

    point_pass = bool(
        final_metrics["full_2022"]["gain"] > 0.0
        and final_metrics["full_2022"]["positive_month_fraction"] >= (6.0 / 7.0)
        and final_metrics["late_2023"]["gain"] > 0.0
        and final_metrics["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and final_metrics["full_2024"]["gain"] > 0.0
        and final_metrics["full_2024"]["positive_month_fraction"] >= 0.625
        and min(item["worst_month_gain"] for item in final_metrics.values()) > -2.0
        and min(item["minimum_domain_gain"] for item in final_metrics.values()) >= 0.0
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    selected_rank = ranked_names.index(selected_gate) + 1
    passed = bool(point_pass and robust_pass)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=selected["full_2022"],
        late_2023=selected["late_2023"],
        full_2024=selected["full_2024"],
        parent_full_2024=parents["full_2024"],
        base_v192_full_2024=base["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "total_stack_pass" if passed else "total_stack_reject",
        "selected_gate_from_v195": selected_gate,
        "frozen_weight": weight,
        "gate_count_in_reality_check": len(gates),
        "base_v192_metrics": base_metrics,
        "total_metrics": final_metrics,
        "robustness_2024": robust,
        "selected_gate_2024_rank": selected_rank,
        "best_2024_gate": ranked_names[0],
        "best_2024_gain": float(family_gains[ranked_names[0]]),
        "point_gate_passed": point_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_final_training": passed,
        "parity": parity,
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
        "total_candidate_compared_to_exact_jy_parent": True,
        "all_predeclared_v195_gates_in_reality_check": True,
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
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v192-axes", type=Path, required=True)
    parser.add_argument("--v184-dir", type=Path, required=True)
    parser.add_argument("--v195-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v192_axes,
        args.v184_dir, args.v195_summary, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
