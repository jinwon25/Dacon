"""Audit whether v178 survives removal of the costly v114/Jiyun component.

The v178 source-selected attenuation (0.25) and the remaining v165 weights are
held fixed.  This is a deployment feasibility ablation, not a new optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v178_jy_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V179_DEPLOYABLE_SIGNED_STACK_ABLATION_V1"
SCALE = 0.25
OMITTED_FAMILY = "v114_independent_source_stability_mask_20260823_01"
AXES = ("full_2022", "late_2023", "full_2024")


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    all_weights = load_weights(v165_summary)
    deployable_weights = {
        name: value for name, value in all_weights.items() if name != OMITTED_FAMILY
    }
    if len(deployable_weights) != len(all_weights) - 1:
        raise ValueError("frozen v114 component was not found")
    with np.load(v158_path, allow_pickle=False) as saved:
        base = {name: saved[name].astype(np.float64) for name in AXES}
    directions = {
        "full": {
            name: load_direction(name, base[name], all_weights, library_root)
            for name in AXES
        },
        "without_v114": {
            name: load_direction(name, base[name], deployable_weights, library_root)
            for name in AXES
        },
    }
    details: dict[str, Any] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for variant, variant_directions in directions.items():
        candidates[variant] = {}
        details[variant] = {}
        for axis_name in AXES:
            candidate = apply_direction(
                parents[axis_name], variant_directions[axis_name], SCALE
            )
            candidates[variant][axis_name] = candidate
            details[variant][axis_name] = metrics(
                axes[axis_name], parents[axis_name], candidate
            )

    source_pass = all(
        details["without_v114"][name]["gain"] > 0.0
        and details["without_v114"][name]["positive_month_fraction"] >= 0.75
        and details["without_v114"][name]["worst_month_gain"] > -5.0
        and details["without_v114"][name]["minimum_domain_gain"] >= 0.0
        for name in ("full_2022", "late_2023")
    )
    locked = details["without_v114"]["full_2024"]
    point_pass = bool(
        source_pass
        and locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
        and locked["minimum_domain_gain"] >= 0.0
    )
    robust = None
    robust_pass = False
    if point_pass:
        exact = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        active = exact & np.not_equal(directions["without_v114"]["full_2024"], 0.0)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"],
            candidates["without_v114"]["full_2024"], active,
            [candidates["without_v114"]["full_2024"]],
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "selected_axis.npz",
        parent=parents["full_2024"],
        candidate=candidates["without_v114"]["full_2024"],
        direction=directions["without_v114"]["full_2024"],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "deployable_robust_pass" if point_pass and robust_pass else "reject",
        "scale_frozen_from_v178": SCALE,
        "omitted_family": OMITTED_FAMILY,
        "remaining_weights": deployable_weights,
        "metrics": details,
        "source_gate_passed": bool(source_pass),
        "point_gate_passed": point_pass,
        "robustness": robust,
        "robust_gate_passed": robust_pass,
        "eligible_for_package_implementation": bool(point_pass and robust_pass),
        "parity": parity,
        "restrictions": {
            "weights_or_scale_refit": False,
            "official_train_only": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
